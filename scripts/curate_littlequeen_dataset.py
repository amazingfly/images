
# Load centralized workstation defaults; explicit environment/CLI values win.
import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
for _workspace_root in _WorkspacePath(__file__).resolve().parents:
    if (_workspace_root / "media_workspace").is_dir():
        _workspace_sys.path.insert(0, str(_workspace_root))
        break
from media_workspace.config import apply_environment as _apply_workspace
_apply_workspace()

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEMMA_START = Path(os.environ.get("LTX_REPO", "/home/derek/projects/agentic/ltxVideo")) / "scripts/start_gemma_vision.sh"
DEFAULT_GEMMA_STOP = Path(os.environ.get("LTX_REPO", "/home/derek/projects/agentic/ltxVideo")) / "scripts/stop_gemma_vision.sh"

GEMMA_CURATION_PROMPT = """
Look at this image and judge if it is suitable for a Little Queen LoRA training dataset.

Reject only for these issues:
1. There is more than one little queen / princess / queen-like girl, including a cropped second head, face, crown, or body entering from an image edge.
2. The main little queen is missing.
3. The main little queen is mostly cropped, mostly hidden, or too tiny to train from.

Important duplicate rule: inspect all four image edges and corners. A partial extra
queen-like face, hair mass, crown, torso, or body at an edge counts as a duplicate
queen even if the full body is not visible.

Review, but do not automatically reject, for:
1. The face or body is severely malformed, melted, or unreadable.
2. The character identity is clearly wrong, such as not a young crowned girl in a pink/magenta dress.

Do not reject for harmless monsters, pets, toys, props, wands, magic effects, background changes, pose changes, clothing variation, or minor hand flaws.

Return only this JSON object, with no markdown:
{
  "single_little_queen": true,
  "duplicate_or_second_queen": false,
  "main_character_present": true,
  "mostly_cropped_or_hidden": false,
  "severe_malformed_character": false,
  "wrong_identity": false,
  "decision": "accept",
  "failed_questions": [],
  "notes": "short reason, 16 words or fewer"
}
""".strip()

CURATION_CHECKS = [
    {
        'id': 'single_character',
        'question': (
            "Review if the image may show more than one little queen or queen-like girl, "
            "including a cropped duplicate head/body entering from an edge."
        ),
        'hard_reject': False,
        'pass_labels': [
            "one single anime princess character",
            "only one crowned girl in the image",
            "a single little queen character",
        ],
        'fail_labels': [
            "two anime princess characters",
            "multiple anime girls in the image",
            "a duplicate princess clone",
            "a cropped second princess head at the edge",
        ],
    },
    {
        'id': 'main_character_present',
        'question': (
            "Reject if the main little queen is missing, tiny, mostly hidden, or mostly "
            "outside the image frame."
        ),
        'hard_reject': True,
        'pass_labels': [
            "a clear crowned little princess in a pink dress",
            "a visible anime queen character wearing a crown",
            "a clear single girl character in the center",
        ],
        'fail_labels': [
            "no person visible",
            "a monster or object only with no girl",
            "a person mostly cropped out of frame",
            "a tiny distant character hard to see",
        ],
    },
    {
        'id': 'coherent_character',
        'question': (
            "Review if the queen's face or body looks badly broken, melted, malformed, "
            "or unreadable."
        ),
        'hard_reject': False,
        'pass_labels': [
            "clean coherent anime character illustration",
            "clear cute anime face and body",
            "well drawn princess character",
        ],
        'fail_labels': [
            "deformed broken anime face",
            "malformed character anatomy",
            "melted distorted illustration",
            "bad broken hands and face",
        ],
    },
]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', str(path))
    ]


def find_images(images_dir):
    images_dir = resolve_path(images_dir)
    return sorted(images_dir.glob('image_*.png'), key=natural_key)


def load_clip(model_name, local_files_only, device):
    logger.info("Loading CLIP curation model: %s", model_name)
    processor = CLIPProcessor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = CLIPModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return processor, model


def http_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST' if payload is not None else 'GET',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f"Gemma vision server returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Gemma vision server at {url}: {exc}") from exc


def prepare_gemma_image(path, max_edge=512):
    guessed_type = mimetypes.guess_type(path.name)[0] or "image/png"
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    mime_type = "image/jpeg" if guessed_type.startswith("image/") else guessed_type
    return mime_type, buffer.getvalue()


class GemmaVisionClient:
    def __init__(self, base_url, start_command=None, stop_command=None, timeout=180):
        self.base_url = base_url.rstrip("/")
        self.start_command = Path(start_command) if start_command else None
        self.stop_command = Path(stop_command) if stop_command else None
        self.timeout = timeout
        self.started_server = False

    def ensure_ready(self):
        if self._healthy():
            return
        if self.start_command is None:
            raise RuntimeError(f"Gemma vision server is not reachable at {self.base_url}")
        logger.info("Starting Gemma vision server with %s", self.start_command)
        subprocess.run([str(self.start_command)], check=True)
        self.started_server = True
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self._healthy():
                return
            time.sleep(1)
        raise RuntimeError(f"Gemma vision server did not become ready at {self.base_url}")

    def stop_if_started(self, keep_running):
        if keep_running or not self.started_server or self.stop_command is None:
            return
        if self.stop_command.is_file():
            logger.info("Stopping Gemma vision server with %s", self.stop_command)
            subprocess.run([str(self.stop_command)], check=False)

    def _healthy(self):
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as response:
                data = json.load(response)
            return data.get("status") == "ok"
        except (OSError, ValueError, urllib.error.URLError):
            return False

    def judge_image(self, image_path):
        mime_type, image_bytes = prepare_gemma_image(image_path)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        correction = ""
        last_error = None
        for _ in range(2):
            payload = {
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': GEMMA_CURATION_PROMPT + correction,
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f"data:{mime_type};base64,{encoded}",
                                },
                            },
                        ],
                    }
                ],
                'temperature': 0.0,
                'top_p': 0.9,
                'max_tokens': 192,
            }
            response = http_json(
                f"{self.base_url}/v1/chat/completions",
                payload=payload,
                timeout=self.timeout,
            )
            try:
                text = response['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Unexpected Gemma vision response: {response}") from exc
            try:
                return parse_gemma_json(text)
            except RuntimeError as exc:
                last_error = exc
                correction = (
                    "\nYour previous answer was not valid JSON. Return only the exact JSON "
                    "object requested, with booleans and no markdown."
                )
        raise last_error or RuntimeError("Gemma vision did not return valid JSON")


def parse_gemma_json(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse Gemma JSON: {text!r}") from exc

    required_bool_keys = [
        'single_little_queen',
        'duplicate_or_second_queen',
        'main_character_present',
        'mostly_cropped_or_hidden',
        'severe_malformed_character',
        'wrong_identity',
    ]
    for key in required_bool_keys:
        if not isinstance(data.get(key), bool):
            raise RuntimeError(f"Gemma JSON missing boolean {key}: {data}")

    decision = data.get('decision')
    if decision not in {'accept', 'review', 'reject'}:
        raise RuntimeError(f"Gemma JSON has invalid decision: {data}")

    if not isinstance(data.get('failed_questions'), list):
        data['failed_questions'] = []
    data['notes'] = str(data.get('notes', '')).strip()
    return data


def score_labels(image, labels, processor, model, device):
    inputs = processor(
        text=labels,
        images=image,
        return_tensors='pt',
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0)

    return {
        label: {
            'probability': round(float(prob), 4),
            'logit': round(float(logit), 4),
        }
        for label, prob, logit in zip(labels, probs, logits)
    }


def evaluate_check(image, check, processor, model, device, reject_margin, review_margin):
    labels = check['pass_labels'] + check['fail_labels']
    scores = score_labels(image, labels, processor, model, device)

    pass_best_label = max(
        check['pass_labels'],
        key=lambda label: scores[label]['probability'],
    )
    fail_best_label = max(
        check['fail_labels'],
        key=lambda label: scores[label]['probability'],
    )
    pass_score = scores[pass_best_label]['probability']
    fail_score = scores[fail_best_label]['probability']
    margin = round(fail_score - pass_score, 4)

    failed = margin >= reject_margin
    needs_review = not failed and margin >= -review_margin

    return {
        'id': check['id'],
        'question': check['question'],
        'pass': not failed,
        'needs_review': needs_review,
        'hard_reject': check['hard_reject'],
        'best_pass_label': pass_best_label,
        'best_fail_label': fail_best_label,
        'pass_score': pass_score,
        'fail_score': fail_score,
        'fail_minus_pass': margin,
        'scores': scores,
    }


def image_basic_stats(image):
    width, height = image.size
    return {
        'width': width,
        'height': height,
        'aspect_ratio': round(width / height, 4),
    }


def character_region_check(image, min_region_area_ratio):
    array = np.asarray(image)
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]

    pink = (red > 150) & (blue > 100) & (red > green + 25) & (blue > green + 10)
    gold = (red > 150) & (green > 90) & (blue < 120) & (red > blue + 50)
    skin = (red > 145) & (green > 80) & (blue > 60) & (red > green + 10) & (green > blue - 10)
    brown = (
        (red > 65)
        & (red < 190)
        & (green > 35)
        & (green < 135)
        & (blue > 20)
        & (blue < 120)
        & (red > green + 8)
        & (green >= blue - 5)
    )
    mask = (pink | gold | skin | brown).astype('uint8') * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    height, width = mask.shape
    image_area = width * height
    components = []
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        area_ratio = area / image_area
        if area_ratio < min_region_area_ratio:
            continue

        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        components.append({
            'area_ratio': round(area_ratio, 4),
            'bbox': [x, y, component_width, component_height],
            'touches_edge': (
                x <= 3
                or y <= 3
                or x + component_width >= width - 3
                or y + component_height >= height - 3
            ),
        })

    components.sort(key=lambda component: component['area_ratio'], reverse=True)
    failed = len(components) >= 2
    return {
        'id': 'disconnected_character_regions',
        'question': (
            "Reject if the image has two or more separated large regions matching the "
            "queen's hair/skin/crown/dress colors, which often means a duplicate or "
            "cropped second queen."
        ),
        'pass': not failed,
        'needs_review': False,
        'hard_reject': True,
        'min_region_area_ratio': min_region_area_ratio,
        'region_count': len(components),
        'regions': components,
    }


def gemma_boolean_check(gemma_result, key, expected, check_id, question, hard_reject):
    actual = bool(gemma_result[key])
    passed = actual is expected
    return {
        'id': check_id,
        'question': question,
        'pass': passed,
        'needs_review': (not passed and not hard_reject),
        'hard_reject': hard_reject,
        'gemma_key': key,
        'expected': expected,
        'actual': actual,
    }


def gemma_decision_check(gemma_result):
    return {
        'id': 'gemma_decision',
        'question': "Gemma's overall decision from the minimal reject/review checklist.",
        'pass': gemma_result['decision'] == 'accept',
        'needs_review': gemma_result['decision'] == 'review',
        'hard_reject': gemma_result['decision'] == 'reject',
        'decision': gemma_result['decision'],
        'failed_questions': gemma_result.get('failed_questions', []),
        'notes': gemma_result.get('notes', ''),
        'raw': gemma_result,
    }


def checks_from_gemma_result(gemma_result):
    return [
        gemma_boolean_check(
            gemma_result,
            'duplicate_or_second_queen',
            False,
            'gemma_duplicate_or_second_queen',
            "Reject if Gemma sees a duplicate or second queen-like character.",
            True,
        ),
        gemma_boolean_check(
            gemma_result,
            'main_character_present',
            True,
            'gemma_main_character_present',
            "Reject if Gemma says the main little queen is missing.",
            True,
        ),
        gemma_boolean_check(
            gemma_result,
            'mostly_cropped_or_hidden',
            False,
            'gemma_mostly_cropped_or_hidden',
            "Reject if Gemma says the main queen is mostly cropped, hidden, or too tiny.",
            True,
        ),
        gemma_boolean_check(
            gemma_result,
            'single_little_queen',
            True,
            'gemma_single_little_queen',
            "Review if Gemma is not confident there is one clear little queen.",
            False,
        ),
        gemma_boolean_check(
            gemma_result,
            'severe_malformed_character',
            False,
            'gemma_severe_malformed_character',
            "Review if Gemma sees severe malformation.",
            False,
        ),
        gemma_boolean_check(
            gemma_result,
            'wrong_identity',
            False,
            'gemma_wrong_identity',
            "Review if Gemma says the identity is clearly wrong.",
            False,
        ),
        gemma_decision_check(gemma_result),
    ]


def decide_from_checks(checks):
    failed_hard_checks = [
        check['id']
        for check in checks
        if not check['pass'] and check['hard_reject']
    ]
    review_checks = [
        check['id']
        for check in checks
        if check['needs_review'] or (not check['pass'] and not check['hard_reject'])
    ]

    if failed_hard_checks:
        decision = 'reject'
    elif review_checks:
        decision = 'review'
    else:
        decision = 'accept'

    return decision, failed_hard_checks, review_checks


def get_gemma_decision(checks):
    for check in checks:
        if check.get('id') == 'gemma_decision':
            return check.get('decision')
    return None


def apply_gemma_heuristic_review_policy(record):
    if record.get('judge') != 'gemma':
        return False
    if record.get('decision') != 'reject':
        return False

    reject_reasons = record.get('reject_reasons', [])
    if reject_reasons != ['disconnected_character_regions']:
        return False
    if get_gemma_decision(record.get('checks', [])) != 'accept':
        return False

    record['decision'] = 'review'
    record['reject_reasons'] = []
    review_reasons = record.setdefault('review_reasons', [])
    if 'heuristic_only_duplicate_region' not in review_reasons:
        review_reasons.append('heuristic_only_duplicate_region')
    record['policy_note'] = (
        "Disconnected-region heuristic flagged this image, but Gemma accepted it; "
        "manual review required."
    )
    return True


def build_record(image_path, caption_path, caption, decision, reject_reasons, review_reasons, checks, stats, judge):
    record = {
        'image_file': str(image_path),
        'caption_file': str(caption_path) if caption_path.exists() else None,
        'caption': caption,
        'decision': decision,
        'reject_reasons': reject_reasons,
        'review_reasons': review_reasons,
        'checks': checks,
        'image_stats': stats,
        'judge': judge,
    }
    apply_gemma_heuristic_review_policy(record)
    return record


def build_error_record(image_path, exc, judge):
    caption_path = image_path.with_suffix('.txt')
    caption = caption_path.read_text(encoding='utf-8').strip() if caption_path.exists() else None
    stats = None
    try:
        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert('RGB')
            stats = image_basic_stats(image)
    except Exception:
        stats = {'error': 'could not read image stats'}

    error_text = f"{exc.__class__.__name__}: {exc}"
    return build_record(
        image_path=image_path,
        caption_path=caption_path,
        caption=caption,
        decision='review',
        reject_reasons=[],
        review_reasons=['curation_error'],
        checks=[
            {
                'id': 'curation_error',
                'question': 'Review if the curation judge failed for this image.',
                'pass': False,
                'needs_review': True,
                'hard_reject': False,
                'error': error_text,
            }
        ],
        stats=stats,
        judge=judge,
    )


def classify_image_clip(
    image_path,
    processor,
    model,
    device,
    reject_margin,
    review_margin,
    min_region_area_ratio,
):
    with Image.open(image_path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert('RGB')
        checks = [character_region_check(image, min_region_area_ratio)]
        checks.extend(
            evaluate_check(
                image,
                check,
                processor,
                model,
                device,
                reject_margin,
                review_margin,
            )
            for check in CURATION_CHECKS
        )
        stats = image_basic_stats(image)

    caption_path = image_path.with_suffix('.txt')
    caption = caption_path.read_text(encoding='utf-8').strip() if caption_path.exists() else None
    decision, reject_reasons, review_reasons = decide_from_checks(checks)

    return build_record(
        image_path,
        caption_path,
        caption,
        decision,
        reject_reasons,
        review_reasons,
        checks,
        stats,
        'clip',
    )


def classify_image_gemma(image_path, gemma_client, min_region_area_ratio):
    with Image.open(image_path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert('RGB')
        checks = [character_region_check(image, min_region_area_ratio)]
        stats = image_basic_stats(image)

    gemma_result = gemma_client.judge_image(image_path)
    checks.extend(checks_from_gemma_result(gemma_result))

    caption_path = image_path.with_suffix('.txt')
    caption = caption_path.read_text(encoding='utf-8').strip() if caption_path.exists() else None
    decision, reject_reasons, review_reasons = decide_from_checks(checks)

    return build_record(
        image_path,
        caption_path,
        caption,
        decision,
        reject_reasons,
        review_reasons,
        checks,
        stats,
        'gemma',
    )


def copy_to_bucket(record, output_dir):
    bucket_dir = output_dir / record['decision']
    bucket_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(record['image_file'])
    shutil.copy2(image_path, bucket_dir / image_path.name)
    if record['caption_file']:
        caption_path = Path(record['caption_file'])
        shutil.copy2(caption_path, bucket_dir / caption_path.name)


def write_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_suffix(report_path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_path, report_path)


def normalize_record_image_path(record):
    image_file = record.get('image_file')
    if not image_file:
        return None
    return str(resolve_path(image_file).resolve())


def load_existing_report(output_path, expected_judge, resume):
    if not resume or not output_path.exists():
        return None, []

    with output_path.open('r', encoding='utf-8') as f:
        report = json.load(f)

    if report.get('judge') != expected_judge:
        logger.warning(
            "Existing report judge is %s, requested %s; starting a new report.",
            report.get('judge'),
            expected_judge,
        )
        return None, []

    records = report.get('records', [])
    migrated = 0
    for record in records:
        if apply_gemma_heuristic_review_policy(record):
            migrated += 1
    if migrated:
        logger.info("Migrated %s heuristic-only Gemma reject(s) to review.", migrated)

    logger.info("Resuming from existing report with %s completed record(s).", len(records))
    return report, records


def summarize(records):
    summary = {'accept': 0, 'review': 0, 'reject': 0}
    for record in records:
        summary[record['decision']] += 1
    return summary


def build_report_questions(judge):
    questions = [
        {
            'id': 'disconnected_character_regions',
            'question': (
                "Reject if the image has two or more separated large regions matching "
                "the queen's hair/skin/crown/dress colors."
            ),
            'hard_reject': True,
            'pass_examples': ['one connected queen-colored character region'],
            'fail_examples': ['main queen plus separate cropped queen head at the edge'],
        }
    ]
    if judge == 'gemma':
        questions.extend([
            {
                'id': 'gemma_duplicate_or_second_queen',
                'question': "Reject if Gemma sees a duplicate or second queen-like character.",
                'hard_reject': True,
            },
            {
                'id': 'gemma_main_character_present',
                'question': "Reject if Gemma says the main little queen is missing.",
                'hard_reject': True,
            },
            {
                'id': 'gemma_mostly_cropped_or_hidden',
                'question': "Reject if Gemma says the main queen is mostly cropped, hidden, or too tiny.",
                'hard_reject': True,
            },
            {
                'id': 'gemma_single_little_queen',
                'question': "Review if Gemma is not confident there is one clear little queen.",
                'hard_reject': False,
            },
            {
                'id': 'gemma_severe_malformed_character',
                'question': "Review if Gemma sees severe malformation.",
                'hard_reject': False,
            },
            {
                'id': 'gemma_wrong_identity',
                'question': "Review if Gemma says the identity is clearly wrong.",
                'hard_reject': False,
            },
        ])
    else:
        questions.extend(
            {
                'id': check['id'],
                'question': check['question'],
                'hard_reject': check['hard_reject'],
                'pass_examples': check['pass_labels'],
                'fail_examples': check['fail_labels'],
            }
            for check in CURATION_CHECKS
        )
    return questions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curate Little Queen LoRA dataset images with Gemma vision or CLIP checks."
    )
    parser.add_argument(
        '--images-dir',
        default='scripts/output_lora_littlequeen_dataset',
        help='Directory containing generated image_*.png files.',
    )
    parser.add_argument(
        '--output',
        default='scripts/output_lora_littlequeen_dataset/curation/curation_report.json',
        help='JSON report path.',
    )
    parser.add_argument(
        '--model',
        default='openai/clip-vit-base-patch32',
        help='CLIP model name or local path.',
    )
    parser.add_argument(
        '--judge',
        choices=('gemma', 'clip'),
        default='gemma',
        help='Image judge to use. Gemma vision is the preferred semantic judge.',
    )
    parser.add_argument(
        '--gemma-url',
        default='http://127.0.0.1:8080',
        help='Gemma vision llama-server base URL.',
    )
    parser.add_argument(
        '--gemma-start',
        type=Path,
        default=DEFAULT_GEMMA_START,
        help='Command that starts the Gemma vision server when needed.',
    )
    parser.add_argument(
        '--gemma-stop',
        type=Path,
        default=DEFAULT_GEMMA_STOP,
        help='Command that stops the Gemma vision server when this script started it.',
    )
    parser.add_argument(
        '--keep-gemma-running',
        action='store_true',
        help='Leave Gemma vision running after curation when this script started it.',
    )
    parser.add_argument(
        '--gemma-timeout',
        type=float,
        default=180.0,
        help='Seconds to wait for each Gemma vision response.',
    )
    parser.add_argument(
        '--local-files-only',
        action='store_true',
        help='Only use already cached model files.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of images to check.',
    )
    parser.add_argument(
        '--copy-buckets',
        action='store_true',
        help='Copy images and captions into accept/review/reject subfolders beside the report.',
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start a fresh report instead of skipping images already present in the output JSON.',
    )
    parser.add_argument(
        '--reject-margin',
        type=float,
        default=0.08,
        help='Reject when best fail label probability beats best pass label by this amount.',
    )
    parser.add_argument(
        '--review-margin',
        type=float,
        default=0.04,
        help='Review when best fail label is within this amount of best pass label.',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='Torch device, normally cpu for this project.',
    )
    parser.add_argument(
        '--min-region-area-ratio',
        type=float,
        default=0.045,
        help='Minimum disconnected queen-colored region size for hard duplicate rejection.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images = find_images(args.images_dir)
    if args.limit is not None:
        images = images[:max(0, args.limit)]
    if not images:
        raise FileNotFoundError(f"No image_*.png files found in {resolve_path(args.images_dir)}")

    output_path = resolve_path(args.output)
    resume = not args.no_resume
    existing_report, records = load_existing_report(output_path, args.judge, resume)
    completed_paths = {
        normalize_record_image_path(record)
        for record in records
        if normalize_record_image_path(record)
    }
    pending_images = [
        image_path
        for image_path in images
        if str(image_path.resolve()) not in completed_paths
    ]

    if existing_report is not None:
        report = existing_report
        report['updated_at'] = datetime.now().isoformat(timespec='seconds')
        report['images_dir'] = str(resolve_path(args.images_dir))
        report['judge'] = args.judge
        report['model'] = args.model if args.judge == 'clip' else args.gemma_url
        report['reject_margin'] = args.reject_margin
        report['review_margin'] = args.review_margin
        report['min_region_area_ratio'] = args.min_region_area_ratio
        report['questions'] = build_report_questions(args.judge)
        report['summary'] = summarize(records)
        report['records'] = records
    else:
        report = {
            'version': 1,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'images_dir': str(resolve_path(args.images_dir)),
            'judge': args.judge,
            'model': args.model if args.judge == 'clip' else args.gemma_url,
            'method': (
                'Gemma 4 vision JSON checklist plus disconnected character-region hard reject'
                if args.judge == 'gemma'
                else 'CLIP zero-shot scoring over fixed reject questions plus disconnected character-region hard reject'
            ),
            'reject_margin': args.reject_margin,
            'review_margin': args.review_margin,
            'min_region_area_ratio': args.min_region_area_ratio,
            'questions': build_report_questions(args.judge),
            'summary': summarize(records),
            'records': records,
        }
    report['method'] = (
        'Gemma 4 vision JSON checklist plus disconnected character-region review backstop'
        if args.judge == 'gemma'
        else 'CLIP zero-shot scoring over fixed reject questions plus disconnected character-region hard reject'
    )
    write_report(output_path, report)

    logger.info(
        "Curation scope: %s image(s), %s already complete, %s pending.",
        len(images),
        len(records),
        len(pending_images),
    )

    if not pending_images:
        logger.info("No pending images; report is already up to date.")
        logger.info("Summary: %s", report['summary'])
        return

    processor = None
    model = None
    gemma_client = None
    if args.judge == 'clip':
        processor, model = load_clip(args.model, args.local_files_only, args.device)
    else:
        gemma_client = GemmaVisionClient(
            base_url=args.gemma_url,
            start_command=args.gemma_start,
            stop_command=args.gemma_stop,
            timeout=args.gemma_timeout,
        )
        gemma_client.ensure_ready()

    try:
        for index, image_path in enumerate(pending_images, start=1):
            logger.info(
                "Curating pending %s/%s with %s: %s",
                index,
                len(pending_images),
                args.judge,
                image_path.name,
            )
            try:
                if args.judge == 'clip':
                    record = classify_image_clip(
                        image_path,
                        processor,
                        model,
                        args.device,
                        args.reject_margin,
                        args.review_margin,
                        args.min_region_area_ratio,
                    )
                else:
                    record = classify_image_gemma(
                        image_path,
                        gemma_client,
                        args.min_region_area_ratio,
                    )
            except Exception as exc:
                logger.exception("Curation failed for %s; marking for review and continuing.", image_path.name)
                record = build_error_record(image_path, exc, args.judge)
            records.append(record)
            report['updated_at'] = datetime.now().isoformat(timespec='seconds')
            report['summary'] = summarize(records)
            report['records'] = records
            write_report(output_path, report)
            if args.copy_buckets:
                copy_to_bucket(record, output_path.parent)
            logger.info(
                "%s -> %s (%s)",
                image_path.name,
                record['decision'],
                ", ".join(record['reject_reasons'] or record['review_reasons']) or "clean",
            )
    finally:
        if gemma_client is not None:
            gemma_client.stop_if_started(args.keep_gemma_running)
    report['updated_at'] = datetime.now().isoformat(timespec='seconds')
    report['summary'] = summarize(records)
    report['records'] = records
    write_report(output_path, report)

    logger.info("Wrote curation report: %s", output_path)
    logger.info("Summary: %s", report['summary'])


if __name__ == "__main__":
    main()
