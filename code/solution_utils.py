"""Feature extraction shared by training and frozen inference.

The implementation deliberately depends only on NumPy and the supplied inputs.
It never opens reference-answer files.
"""

from __future__ import annotations

import ast
from fractions import Fraction
import math
import re
from collections import Counter

import numpy as np

from common import parse_answer


FEATURE_VERSION = 1


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_WORD_RE = re.compile(r"[A-Za-z]+")
_NUMERIC_FRAC_RE = re.compile(
    r"\\(?:d)?frac\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*"
    r"\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
)
_TEXT_COMMAND_RE = re.compile(r"\\text\s*\{[^{}]*\}")


def parsed_answers(row):
    """Return the scorer-compatible parsed answer for each candidate."""
    return [parse_answer(candidate["text"]) for candidate in row["candidates"]]


def majority_choice(row):
    """Choose the earliest candidate in the largest valid answer group."""
    answers = parsed_answers(row)
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        return 0
    best_answer = max(counts, key=lambda answer: (counts[answer], -answers.index(answer)))
    return answers.index(best_answer)


def _safe_float(answer):
    if answer is None:
        return None
    try:
        numerator, separator, denominator = answer.partition("/")
        value = float(numerator) / float(denominator) if separator else float(numerator)
        return value if math.isfinite(value) else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _surface_features(question, text, answer, group_count, max_group_count, index):
    """Small, auditable features derived from the question and supplied trace."""
    lower = text.lower()
    words = _WORD_RE.findall(lower)
    q_numbers = {token.replace(",", "") for token in _NUMBER_RE.findall(question)}
    t_numbers = [token.replace(",", "") for token in _NUMBER_RE.findall(text)]
    covered = len(q_numbers.intersection(t_numbers)) / max(1, len(q_numbers))
    value = _safe_float(answer)
    answer_in_question = float(answer is not None and answer in q_numbers)
    log_abs_answer = math.log1p(abs(value)) if value is not None else 0.0
    is_integer = float(value is not None and value.is_integer())

    position = [float(index == i) for i in range(4)]
    return np.asarray(
        [
            float(answer is not None),
            group_count / 4.0,
            float(group_count == max_group_count),
            float(max_group_count == 1),
            math.log1p(len(text)),
            math.log1p(len(words)),
            math.log1p(text.count("\n")),
            math.log1p(len(t_numbers)),
            covered,
            answer_in_question,
            log_abs_answer,
            is_integer,
            float("####" in text),
            float("\\boxed" in text),
            float("final answer" in lower),
            float("therefore" in lower or "thus" in lower),
            math.log1p(lower.count("step")),
            math.log1p(text.count("=")),
            math.log1p(sum(text.count(op) for op in ("+", "-", "*", "/"))),
            *position,
        ],
        dtype=np.float32,
    )


def _evaluate_arithmetic(node):
    """Evaluate an already parsed arithmetic AST without calling Python eval."""
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_arithmetic(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow) and abs(right) <= 12:
            return left**right
    raise ValueError("unsupported arithmetic expression")


def _safe_arithmetic_value(expression):
    """Parse the numeric part of a simple equation side, if it is auditable."""
    expression = _NUMERIC_FRAC_RE.sub(r"(\1/\2)", expression)
    expression = _TEXT_COMMAND_RE.sub("", expression)
    for old, new in (
        (r"\times", "*"),
        (r"\cdot", "*"),
        (r"\div", "/"),
        ("×", "*"),
        ("÷", "/"),
        ("−", "-"),
        ("^", "**"),
        (",", ""),
        ("$", ""),
        ("{", "("),
        ("}", ")"),
    ):
        expression = expression.replace(old, new)
    expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expression)
    expression = re.sub(r"\\[%$#&]", "", expression).strip()
    start = re.search(r"[-+(]?\d", expression)
    if not start:
        return None
    expression = re.sub(r"\\[A-Za-z]+", "", expression[start.start() :])
    if re.search(r"[A-Za-z_]|[^\d.()+*/%\s-]", expression):
        return None
    try:
        value = float(_evaluate_arithmetic(ast.parse(expression, mode="eval")))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    return value if math.isfinite(value) and abs(value) < 1e15 else None


def _arithmetic_features(text, answer):
    """Measure whether explicit numeric equation chains are self-consistent."""
    comparisons = valid = invalid = 0
    errors = []
    last_value = None
    for line in text.splitlines():
        if "=" not in line:
            continue
        values = [_safe_arithmetic_value(part) for part in line.split("=")]
        for left, right in zip(values, values[1:]):
            if left is None or right is None:
                continue
            comparisons += 1
            error = abs(left - right) / max(1.0, abs(left), abs(right))
            errors.append(error)
            if error < 1e-8:
                valid += 1
            else:
                invalid += 1
            last_value = right

    answer_value = _safe_float(answer)
    answer_matches_last = float(
        answer_value is not None
        and last_value is not None
        and abs(answer_value - last_value)
        <= 1e-8 * max(1.0, abs(answer_value), abs(last_value))
    )
    return np.asarray(
        [
            math.log1p(comparisons),
            math.log1p(valid),
            math.log1p(invalid),
            valid / max(1, comparisons),
            invalid / max(1, comparisons),
            math.log1p(sum(errors)),
            math.log1p(max(errors, default=0.0)),
            float(comparisons == 0),
            answer_matches_last,
        ],
        dtype=np.float32,
    )


def _safe_arithmetic_prefix(expression):
    """Evaluate a numeric expression followed by ordinary prose or units.

    Candidate traces often write ``12 / 3 = 4 people``.  The stricter parser
    above intentionally rejects the unit suffix; this companion parser keeps
    the auditable arithmetic prefix while still rejecting algebraic variables
    such as ``24t``.
    """
    expression = expression.replace("**", "")
    expression = _NUMERIC_FRAC_RE.sub(r"(\1/\2)", expression)
    expression = _TEXT_COMMAND_RE.sub("", expression)
    for old, new in (
        (r"\times", "*"), (r"\cdot", "*"), (r"\div", "/"),
        (r"\left", ""), (r"\right", ""), (r"\boxed", ""),
        ("×", "*"), ("÷", "/"), ("−", "-"), ("^", "**"),
        (",", ""), ("$", ""), ("{", "("), ("}", ")"),
    ):
        expression = expression.replace(old, new)
    expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expression)
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*\\?%", r"(\1/100)", expression)
    expression = re.sub(r"\\[%$#&]", "", expression)
    start = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", expression)
    if not start:
        return None
    begin = start.start()
    while begin > 0 and expression[begin - 1] in "(+- ":
        begin -= 1
    tail = expression[begin:]
    match = re.match(r"[\d.()+*/%\s-]+", tail)
    if not match:
        return None
    candidate = match.group().strip()
    next_character = tail[match.end():match.end() + 1]
    # An adjacent letter denotes an algebraic variable (24t). A separated
    # letter starts a unit/prose suffix (24 miles), which is safe to ignore.
    if next_character and next_character.isalpha() and match.group() and not match.group()[-1].isspace():
        return None
    candidate = candidate.rstrip(". ")
    try:
        value = float(_evaluate_arithmetic(ast.parse(candidate, mode="eval")))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    return value if math.isfinite(value) and abs(value) < 1e15 else None


def _enhanced_arithmetic_features(text, answer):
    """Arithmetic checks that tolerate prose unit suffixes and rounding."""
    comparisons = exact = approximate = invalid = severe = 0
    errors = []
    last_value = None
    normalized = text.replace(r"\approx", "=").replace("≈", "=")
    for line in normalized.splitlines():
        if "=" not in line:
            continue
        values = [_safe_arithmetic_prefix(part) for part in line.split("=")]
        for left, right in zip(values, values[1:]):
            if left is None or right is None:
                continue
            comparisons += 1
            error = abs(left - right) / max(1.0, abs(left), abs(right))
            errors.append(error)
            exact += int(error < 1e-8)
            approximate += int(error < 1e-3)
            invalid += int(error >= 1e-3)
            severe += int(error >= 5e-2)
            last_value = right
    answer_value = _safe_float(answer)
    answer_matches_last = float(
        answer_value is not None and last_value is not None
        and abs(answer_value - last_value) <= 1e-3 * max(1.0, abs(answer_value), abs(last_value))
    )
    maximum = max(errors, default=0.0)
    return np.asarray(
        [
            math.log1p(comparisons), math.log1p(exact), math.log1p(approximate),
            math.log1p(invalid), math.log1p(severe), invalid / max(1, comparisons),
            severe / max(1, comparisons), math.log1p(sum(errors)), math.log1p(maximum),
            math.log10(max(maximum, 1e-12)) if comparisons else 0.0,
            float(comparisons == 0), answer_matches_last,
        ],
        dtype=np.float32,
    )


def _numeric_features(question, answers):
    """Describe each final answer relative to its peers and question numbers."""
    fractions = [Fraction(value) if value is not None else None for value in answers]
    unique = sorted({value for value in fractions if value is not None})
    question_values = []
    for token in _NUMBER_RE.findall(question):
        try:
            question_values.append(float(Fraction(token.replace(",", ""))))
        except (ValueError, ZeroDivisionError, OverflowError):
            pass

    closure = list(question_values)
    for left in question_values:
        for right in question_values:
            for value in (left + right, left - right, left * right):
                if math.isfinite(value):
                    closure.append(value)
            if right:
                value = left / right
                if math.isfinite(value):
                    closure.append(value)

    numeric_unique = [float(value) for value in unique]
    signed_logs = [math.copysign(math.log1p(abs(value)), value) for value in numeric_unique]
    median_log = float(np.median(signed_logs)) if signed_logs else 0.0

    def close(left, right):
        return abs(left - right) <= 1e-8 * max(1.0, abs(left), abs(right))

    rows = []
    for value in fractions:
        feature = np.zeros(31, dtype=np.float32)
        if value is not None:
            rank = unique.index(value)
            count = len(unique)
            numeric = float(value)
            signed_log = math.copysign(math.log1p(abs(numeric)), numeric)
            feature[0] = 1.0
            feature[count] = 1.0
            feature[5 + (count - 1) * 4 + rank] = 1.0
            feature[21] = rank / max(count - 1, 1)
            feature[22] = signed_log - median_log
            feature[23] = abs(signed_log - median_log)
            feature[24] = float(rank == 0)
            feature[25] = float(rank == count - 1)
            feature[26] = float(any(close(numeric, item) for item in question_values))
            feature[27] = float(any(close(numeric, item) for item in closure))
            feature[28] = float(numeric >= 0)
            feature[29] = float(value.denominator == 1)
            max_question = max((abs(item) for item in question_values), default=0.0)
            feature[30] = math.log1p(abs(numeric)) - math.log1p(max_question)
        rows.append(feature)
    return rows


def build_feature_blocks(bank, embeddings):
    """Build answer-vote, surface, and relative full-trace embedding blocks."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[:2] != (len(bank), 4):
        raise ValueError("Embedding rows do not align with the candidate bank")

    surface_rows = []
    arithmetic_rows = []
    enhanced_arithmetic_rows = []
    numeric_rows = []
    valid_mask = np.zeros((len(bank), 4), dtype=bool)
    vote = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        answers = parsed_answers(row)
        numeric_rows.extend(_numeric_features(row["question"], answers))
        counts = Counter(answer for answer in answers if answer is not None)
        max_count = max(counts.values(), default=0)
        for candidate_index, (candidate, answer) in enumerate(zip(row["candidates"], answers)):
            count = counts.get(answer, 0) if answer is not None else 0
            valid_mask[row_index, candidate_index] = answer is not None
            vote[row_index, candidate_index] = count / 4.0
            surface_rows.append(
                _surface_features(
                    row["question"], candidate["text"], answer, count, max_count, candidate_index
                )
            )
            arithmetic_rows.append(_arithmetic_features(candidate["text"], answer))
            enhanced_arithmetic_rows.append(
                _enhanced_arithmetic_features(candidate["text"], answer)
            )

    # Subtracting the question-wise mean focuses the linear verifier on how a
    # candidate differs from the other three traces for the same question.
    relative_embeddings = embeddings - embeddings.mean(axis=1, keepdims=True)
    embedding_norms = np.linalg.norm(relative_embeddings, axis=2, keepdims=True)
    relative_embeddings = relative_embeddings / np.maximum(embedding_norms, 1e-6)

    return {
        "surface": np.stack(surface_rows).reshape(len(bank), 4, -1),
        "arithmetic": np.stack(arithmetic_rows).reshape(len(bank), 4, -1),
        "arithmetic_v2": np.stack(enhanced_arithmetic_rows).reshape(len(bank), 4, -1),
        "numeric": np.stack(numeric_rows).reshape(len(bank), 4, -1),
        "embedding": relative_embeddings,
        "vote": vote,
        "valid": valid_mask,
    }


def select_matrix(blocks, feature_set):
    """Flatten the requested feature blocks to [4*N, D]."""
    chosen = []
    if feature_set in ("surface", "combined"):
        chosen.append(blocks["surface"])
    if feature_set in ("embedding", "combined"):
        chosen.append(blocks["embedding"])
    if feature_set in ("arithmetic", "surface_arithmetic"):
        chosen.append(blocks["arithmetic"])
    if feature_set == "arithmetic_v2":
        chosen.append(blocks["arithmetic_v2"])
    if feature_set == "numeric":
        chosen.append(blocks["numeric"])
    if feature_set == "surface_arithmetic":
        chosen.append(blocks["surface"])
    if not chosen:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return np.concatenate(chosen, axis=2).reshape(-1, sum(block.shape[2] for block in chosen))


def normalize_features(matrix, mean=None, scale=None):
    matrix = np.asarray(matrix, dtype=np.float32)
    if mean is None:
        mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    if scale is None:
        scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
    return (matrix - mean) / scale, mean, scale


def choices_from_scores(scores, valid_mask):
    scores = np.asarray(scores).reshape(len(valid_mask), 4).copy()
    scores[~valid_mask] = -np.inf
    no_valid = ~valid_mask.any(axis=1)
    if no_valid.any():
        scores[no_valid] = 0.0
    return [int(value) for value in np.argmax(scores, axis=1)]


def choices_from_answer_groups(
    bank, scores, temperature=1.0, reducer="logsumexp", count_bonus=0.0
):
    """Aggregate candidate evidence by parsed numeric answer before choosing.

    ``logsumexp`` matches the group likelihood used by the listwise trainer.
    ``max`` reproduces ordinary candidate ranking, and ``mean`` is included as
    a fixed diagnostic ablation. ``count_bonus * log(group size)`` provides a
    scale-stable way to add corroboration from candidates with the same answer.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = np.asarray(scores, dtype=np.float64).reshape(len(bank), 4)
    choices = []
    for row, row_scores in zip(bank, scores):
        answers = parsed_answers(row)
        groups = {}
        for index, answer in enumerate(answers):
            if answer is not None:
                groups.setdefault(answer, []).append(index)
        if not groups:
            choices.append(0)
            continue

        group_scores = []
        for answer, indices in groups.items():
            values = row_scores[indices]
            if reducer == "max":
                value = float(values.max())
            elif reducer == "mean":
                value = float(values.mean())
            elif reducer == "logsumexp":
                scaled = values / temperature
                peak = float(scaled.max())
                value = temperature * (peak + math.log(float(np.exp(scaled - peak).sum())))
            else:
                raise ValueError(f"Unknown answer-group reducer: {reducer}")
            value += count_bonus * math.log(len(indices))
            group_scores.append((value, answer, indices))

        # Dict insertion order and max's first-on-ties behavior preserve the
        # earliest candidate as the deterministic tie break.
        _, _, best_indices = max(group_scores, key=lambda item: item[0])
        best_candidate = max(best_indices, key=lambda index: (row_scores[index], -index))
        choices.append(int(best_candidate))
    return choices


# Bounded read-once expression evidence used by the V12 verifier.  This code
# intentionally stays in the NumPy-only inference module: it enumerates small
# arithmetic graphs over quantities supplied in each question and never calls
# eval or executes candidate text.
_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}


def _read_once_quantities(text):
    result = []
    for token in _NUMBER_RE.findall(text.lower()):
        try:
            result.append(Fraction(token.replace(",", "")))
        except (ValueError, ZeroDivisionError):
            pass
    words = text.lower().replace("-", " ").split()
    result.extend(Fraction(_WORD_NUMBERS[word]) for word in words if word in _WORD_NUMBERS)
    return result[:8]


def _read_once_expected_operations(question):
    q = question.lower()
    has = lambda phrases: any(phrase in q for phrase in phrases)
    return (
        int(has(("total", "altogether", "combined", "in all", "more", "additional", "sum")))
        | (int(has(("remaining", "left", "rest", "difference", "less", "fewer", "discount", "change", "ago"))) << 1)
        | (int(has(("each", "every", "times", "twice", "double", "dozen", "pair", "weekly", "monthly", "percent"))) << 2)
        | (int(has(("equally", "evenly", "average", "per person", "split", "divided", "rate"))) << 3)
    )


def _read_once_features_for_row(row):
    values = _read_once_quantities(row["question"])
    answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
    targets = [Fraction(answer) if answer is not None else None for answer in answers]
    target_values = [target for target in targets if target is not None]
    expected = _read_once_expected_operations(row["question"])
    n = len(values)
    states = {(value, 1 << index, 0, 0) for index, value in enumerate(values)}
    all_states = set(states)

    def safe(value):
        return abs(value) <= 10**10 and value.denominator <= 100000

    def priority(state):
        value, mask, _, steps = state
        if not target_values:
            distance = 0.0
        else:
            signed = math.copysign(math.log1p(abs(float(value))), float(value))
            distance = min(
                abs(signed - math.copysign(math.log1p(abs(float(target))), float(target)))
                for target in target_values
            )
        return (distance, -mask.bit_count(), steps, value.denominator)

    for step in range(1, min(4, n)):
        new_states = set()
        for value, mask, operations, _ in states:
            for index, number in enumerate(values):
                if mask & (1 << index):
                    continue
                new_mask = mask | (1 << index)
                candidates = (
                    (value + number, 1), (value - number, 2),
                    (number - value, 2), (value * number, 4),
                )
                if number:
                    candidates += ((value / number, 8),)
                if value:
                    candidates += ((number / value, 8),)
                for result, operation in candidates:
                    if safe(result):
                        new_states.add((result, new_mask, operations | operation, step))
        if len(new_states) > 800:
            exact = [state for state in new_states if state[0] in target_values]
            new_states = set(sorted(new_states, key=priority)[:800]).union(exact)
        states = new_states
        all_states.update(states)

    rows = []
    full_mask = (1 << n) - 1
    for target in targets:
        feature = np.zeros(24, dtype=np.float32)
        feature[0] = float(target is not None)
        feature[1] = min(n, 8) / 8.0
        if target is not None:
            matches = [state for state in all_states if state[0] == target]
            if matches:
                coverages = [state[1].bit_count() for state in matches]
                steps = [state[3] for state in matches]
                best_coverage = max(coverages)
                best_steps = min(steps)
                masks = [state[2] for state in matches]
                feature[2] = 1.0
                feature[3] = best_coverage / max(1, n)
                feature[4] = float(any(state[1] == full_mask for state in matches))
                feature[5] = best_steps / 5.0
                feature[6] = math.log1p(len(matches))
                feature[7] = float(any((mask & expected) == expected for mask in masks))
                feature[8] = max((mask & expected).bit_count() for mask in masks) / max(1, expected.bit_count())
                feature[9] = float(any(mask == expected for mask in masks))
                for bit in range(4):
                    feature[10 + bit] = float(any(mask & (1 << bit) for mask in masks))
                    feature[14 + bit] = float(bool(expected & (1 << bit)) and any(mask & (1 << bit) for mask in masks))
                feature[18] = float(best_coverage == 1)
                feature[19] = float(best_coverage >= 2)
                feature[20] = float(best_coverage >= 3)
                feature[21] = float(best_coverage >= 4)
                feature[22] = float(best_coverage == n)
                feature[23] = float(best_steps == max(0, best_coverage - 1))
        rows.append(feature)
    return rows


def read_once_expression_features(bank):
    """Return [question, candidate, 24] bounded arithmetic-graph features."""
    return np.stack([
        np.stack(_read_once_features_for_row(row)) for row in bank
    ]).astype(np.float32)


def _trace_close(left, right):
    return abs(left - right) <= 1e-6 * max(1.0, abs(left), abs(right))


def _trace_values(text):
    result = []
    text = _NUMERIC_FRAC_RE.sub(r"(\1/\2)", text)
    for token in _NUMBER_RE.findall(text):
        try:
            result.append(float(Fraction(token.replace(",", ""))))
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    return result


def _trace_normalized_math(text):
    text = _NUMERIC_FRAC_RE.sub(r"(\1/\2)", text)
    for old, new in (
        (r"\times", "*"), (r"\cdot", "*"), (r"\div", "/"),
        ("×", "*"), ("÷", "/"), ("−", "-"), (",", ""),
        (r"\%", "%"), ("{", "("), ("}", ")"),
    ):
        text = text.replace(old, new)
    return text


def _trace_operation_mask(text):
    return (int("+" in text) | (int("-" in text) << 1)
            | (int("*" in text) << 2) | (int("/" in text) << 3))


def _trace_candidate_features(question, candidate):
    text = candidate["text"]
    parsed = parse_answer(text)
    answer = float(Fraction(parsed)) if parsed is not None else None
    question_values = [float(value) for value in _read_once_quantities(question)]
    percent_question = "%" in question or "percent" in question.lower()
    expected = _read_once_expected_operations(question)

    def masks(value):
        result = 0
        for index, question_value in enumerate(question_values[:8]):
            equivalents = (question_value,)
            if percent_question and 0 <= question_value <= 100:
                equivalents += (question_value / 100.0,)
            if any(_trace_close(value, equivalent) for equivalent in equivalents):
                result |= 1 << index
        return result

    provenance, depths = {}, {}
    for value in question_values:
        key = round(value, 8)
        provenance[key] = provenance.get(key, 0) | masks(value)
        depths[key] = 0
    equations = valid = approximate = invalid = severe = 0
    operation_counts = np.zeros(4, dtype=np.float32)
    observed = 0
    last_result = None
    linked_masks, linked_depths, operands_used = [], [], []
    normalized = text.replace(r"\approx", "=").replace("≈", "=")
    for raw_line in normalized.splitlines():
        line = _trace_normalized_math(raw_line)
        if "=" not in line:
            continue
        parts = line.split("=")
        for left, right in zip(parts, parts[1:]):
            left_value = _safe_arithmetic_prefix(left)
            right_value = _safe_arithmetic_prefix(right)
            if left_value is None or right_value is None:
                continue
            equations += 1
            error = abs(left_value - right_value) / max(
                1.0, abs(left_value), abs(right_value)
            )
            valid += int(error < 1e-8)
            approximate += int(error < 1e-3)
            invalid += int(error >= 1e-3)
            severe += int(error >= 5e-2)
            if error >= 1e-3:
                continue
            operands = _trace_values(left)
            operations = _trace_operation_mask(left)
            if not operations or not operands:
                continue
            operands_used.extend(operands)
            observed |= operations
            for bit in range(4):
                operation_counts[bit] += int(bool(operations & (1 << bit)))
            mask, depth = 0, 0
            for operand in operands:
                key = round(operand, 8)
                mask |= provenance.get(key, masks(operand))
                depth = max(depth, depths.get(key, 0))
            result_key = round(right_value, 8)
            provenance[result_key] = provenance.get(result_key, 0) | mask
            depths[result_key] = max(depths.get(result_key, 0), depth + 1)
            last_result = right_value
            if answer is not None and _trace_close(right_value, answer):
                linked_masks.append(mask)
                linked_depths.append(depth + 1)

    answer_mask = max(linked_masks, key=int.bit_count) if linked_masks else 0
    operand_mask = 0
    for operand in operands_used:
        operand_mask |= masks(operand)
    full_mask = (1 << min(len(question_values), 8)) - 1
    lower = question.lower()
    conversions = (
        (percent_question, (100.0,)),
        ("hour" in lower and "minute" in lower, (60.0,)),
        ("week" in lower and "day" in lower, (7.0,)),
        ("dozen" in lower, (12.0,)),
        ("year" in lower and "month" in lower, (12.0,)),
        ("year" in lower and "week" in lower, (52.0,)),
    )
    conversion_features = []
    for required, constants in conversions:
        used = any(any(_trace_close(value, constant) for value in operands_used)
                   for constant in constants)
        conversion_features.extend((float(required and used),
                                    float(required and not used)))
    expected_count = expected.bit_count()
    return np.asarray([
        float(parsed is not None), math.log1p(len(question_values)),
        math.log1p(equations), math.log1p(valid), math.log1p(approximate),
        math.log1p(invalid), math.log1p(severe),
        valid / max(1, equations), invalid / max(1, equations),
        severe / max(1, equations), float(equations == 0),
        float(bool(linked_masks)),
        float(answer is not None and last_result is not None
              and _trace_close(answer, last_result)),
        math.log1p(len(linked_masks)), max(linked_depths, default=0) / 6.0,
        answer_mask.bit_count() / max(1, len(question_values)),
        float(answer_mask == full_mask and full_mask != 0),
        operand_mask.bit_count() / max(1, len(question_values)),
        float(operand_mask == full_mask and full_mask != 0),
        (expected & observed).bit_count() / max(1, expected_count),
        (observed & ~expected).bit_count() / 4.0,
        math.log1p(len(operands_used)),
        *[float(bool(observed & (1 << bit))) for bit in range(4)],
        *[math.log1p(value) for value in operation_counts],
        *[float(bool(answer_mask & (1 << bit))) for bit in range(8)],
        *[float(bool(operand_mask & (1 << bit))) for bit in range(8)],
        *conversion_features,
    ], dtype=np.float32)


def trace_execution_features(bank):
    """Return [question, candidate, 58] executed-equation provenance features."""
    return np.stack([
        np.stack([_trace_candidate_features(row["question"], candidate)
                  for candidate in row["candidates"]])
        for row in bank
    ]).astype(np.float32)


def _program_key(value):
    return round(float(value), 8)


def _program_signed_log(value):
    return math.copysign(math.log1p(abs(value)), value)


def _program_question_numbers(text):
    output = []
    for token in _NUMBER_RE.findall(text):
        try:
            value = float(Fraction(token.replace(",", "")))
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if math.isfinite(value):
            output.append(value)
    return output[:8]


def _bounded_arithmetic_programs(numbers, targets, beam=60):
    n = len(numbers)
    maximum_size = min(n, 5)
    tables = {1 << index: {_program_key(number): 0}
              for index, number in enumerate(numbers)}
    target_logs = [_program_signed_log(value)
                   for value in targets if value is not None]
    hits = {_program_key(value): [] for value in targets if value is not None}
    for size in range(1, maximum_size + 1):
        for mask in range(1, 1 << n):
            if mask.bit_count() != size:
                continue
            if size > 1:
                values = {}
                sub = (mask - 1) & mask
                while sub:
                    other = mask ^ sub
                    if other and sub < other and sub in tables and other in tables:
                        for left, left_ops in tables[sub].items():
                            for right, right_ops in tables[other].items():
                                candidates = (
                                    (left + right, 1), (left - right, 2),
                                    (right - left, 2), (left * right, 4),
                                )
                                for value, operation in candidates:
                                    if math.isfinite(value) and abs(value) <= 1e12:
                                        k = _program_key(value)
                                        values[k] = (values.get(k, 0) | left_ops
                                                     | right_ops | operation)
                                if abs(right) > 1e-10:
                                    value = left / right
                                    if math.isfinite(value) and abs(value) <= 1e12:
                                        k = _program_key(value)
                                        values[k] = (values.get(k, 0) | left_ops
                                                     | right_ops | 8)
                                if abs(left) > 1e-10:
                                    value = right / left
                                    if math.isfinite(value) and abs(value) <= 1e12:
                                        k = _program_key(value)
                                        values[k] = (values.get(k, 0) | left_ops
                                                     | right_ops | 8)
                    sub = (sub - 1) & mask
                if len(values) > beam:
                    def priority(item):
                        value = item[0]
                        distance = min(
                            (abs(_program_signed_log(value) - target)
                             for target in target_logs), default=0.0
                        )
                        integer_bonus = (
                            -.05 if abs(value - round(value)) < 1e-8 else 0.0
                        )
                        return distance + integer_bonus
                    values = dict(sorted(values.items(), key=priority)[:beam])
                tables[mask] = values
            for target_key in hits:
                if target_key in tables.get(mask, {}):
                    hits[target_key].append(
                        (size, tables[mask][target_key], mask)
                    )
    reachable = [value for table in tables.values() for value in table]
    return hits, reachable


def _program_features_for_question(row):
    numbers = _program_question_numbers(row["question"])
    answers = parsed_answers(row)
    targets = []
    for answer in answers:
        try:
            targets.append(float(Fraction(answer)) if answer is not None else None)
        except (ValueError, ZeroDivisionError, OverflowError):
            targets.append(None)
    hits, reachable = _bounded_arithmetic_programs(numbers, targets)
    n = len(numbers)
    full_mask = (1 << n) - 1
    rows = []
    for target in targets:
        feature = np.zeros(19, dtype=np.float32)
        feature[1] = n / 8.0
        if target is None:
            rows.append(feature)
            continue
        feature[0] = 1.0
        target_hits = hits.get(_program_key(target), [])
        feature[2] = float(any(abs(target - value) < 1e-8
                               for value in numbers))
        feature[3] = float(bool(target_hits))
        if target_hits:
            sizes = [item[0] for item in target_hits]
            feature[4] = max(sizes) / max(1, n)
            feature[5] = min(sizes) / max(1, n)
            feature[6] = math.log1p(len(target_hits)) / 4.0
            feature[7] = float(any(item[2] == full_mask for item in target_hits))
            for size in range(1, 6):
                feature[7 + size] = float(size in sizes)
            best = max(target_hits, key=lambda item: (item[0], -item[2]))
            for bit in range(4):
                feature[13 + bit] = float(bool(best[1] & (1 << bit)))
        target_log = _program_signed_log(target)
        feature[17] = min(
            (abs(target_log - _program_signed_log(value))
             for value in reachable), default=20.0
        ) / 10.0
        feature[18] = float(abs(target - round(target)) < 1e-8)
        rows.append(feature)
    return np.stack(rows)


def program_search_features(bank):
    """Return [question, candidate, 19] bounded symbolic-search features."""
    return np.stack([_program_features_for_question(row) for row in bank])
