import hashlib
import hmac
from datetime import timezone

from flask import Blueprint, current_app, request

from database.db import db
from models.codeforces_account import CodeforcesAccount
from models.codeforces_submission import CodeforcesSubmission
from models.practice_session import PracticeSession
from models.problem import Problem


training_export = Blueprint(
    "training_export",
    __name__,
    url_prefix="/api/internal/project-red",
)


def authorized():
    expected = current_app.config.get(
        "PROJECT_RED_EXPORT_TOKEN", ""
    )
    supplied = request.headers.get("Authorization", "")

    if not expected or not supplied.startswith("Bearer "):
        return False

    token = supplied[len("Bearer "):]
    return hmac.compare_digest(token, expected)


def timestamp(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).isoformat()


def user_key(uid, account_id, secret):
    message = f"{uid}:{account_id}".encode()

    return hmac.new(
        secret.encode(),
        message,
        hashlib.sha256,
    ).hexdigest()


def submission_exists(*conditions):
    return db.select(CodeforcesSubmission.id).where(
        CodeforcesSubmission.codeforces_account_id
        == PracticeSession.source_account_id,
        CodeforcesSubmission.problem_id
        == PracticeSession.problem_id,
        CodeforcesSubmission.attribution_status
        == "INDIVIDUAL",
        *conditions,
    ).exists()


@training_export.get("/events")
def export_events():
    if not authorized():
        return {"message": "Unauthorized"}, 401

    secret = current_app.config.get(
        "PROJECT_RED_PSEUDONYM_KEY", ""
    )

    if not secret:
        return {"message": "Export not configured"}, 503

    try:
        after_id = int(request.args.get("after_id", 0))
        limit = int(request.args.get("limit", 200))
    except (TypeError, ValueError):
        return {"message": "Invalid pagination"}, 400

    if after_id < 0 or not 1 <= limit <= 500:
        return {"message": "Invalid pagination"}, 400

    during = (
        CodeforcesSubmission.submitted_at
        >= PracticeSession.started_at,
        CodeforcesSubmission.submitted_at
        <= PracticeSession.finished_at,
    )

    solved = submission_exists(
        *during,
        CodeforcesSubmission.verdict == "OK",
        CodeforcesSubmission.testset == "TESTS",
    )

    attempted = submission_exists(*during)

    previously_solved = submission_exists(
        CodeforcesSubmission.submitted_at
        < PracticeSession.started_at,
        CodeforcesSubmission.verdict == "OK",
        CodeforcesSubmission.testset == "TESTS",
    )

    query = (
        db.select(
            PracticeSession,
            Problem,
            CodeforcesAccount,
            solved.label("solved"),
            attempted.label("attempted"),
            previously_solved.label("previously_solved"),
        )
        .join(
            Problem,
            PracticeSession.problem_id == Problem.id,
        )
        .join(
            CodeforcesAccount,
            PracticeSession.source_account_id
            == CodeforcesAccount.id,
        )
        .where(
            PracticeSession.id > after_id,
            PracticeSession.uid == CodeforcesAccount.uid,
            PracticeSession.status == "FINISHED",
            PracticeSession.finished_at.is_not(None),
            PracticeSession.duration_seconds.between(
                60, 14400
            ),
            Problem.platform == "CODEFORCES",
            CodeforcesAccount.ownership_verified.is_(True),
            CodeforcesAccount.unlinked_at.is_(None),
            CodeforcesAccount.sync_status == "SUCCEEDED",
            CodeforcesAccount.last_synced_at.is_not(None),
            CodeforcesAccount.last_synced_at
            >= PracticeSession.finished_at,
        )
        .order_by(PracticeSession.id.asc())
        .limit(limit + 1)
    )

    rows = db.session.execute(query).all()

    has_more = len(rows) > limit
    rows = rows[:limit]

    events = []

    for session, problem, account, ok, attempt, prior in rows:
        difficulty = (
            problem.difficulty_raw
            if isinstance(problem.difficulty_raw, dict)
            else {}
        )

        rating = difficulty.get("rating")
        if type(rating) is not int or not 800 <= rating <= 3500:
            rating = None

        tags = problem.tags_raw
        if not isinstance(tags, list):
            tags = []

        events.append({
            "event_id": session.id,
            "user_key": user_key(
                session.uid, account.id, secret
            ),
            "problem_external_id": problem.external_id,
            "problem_rating": rating,
            "tags": [
                tag for tag in tags
                if isinstance(tag, str)
            ],
            "started_at": timestamp(session.started_at),
            "finished_at": timestamp(session.finished_at),
            "duration_seconds": session.duration_seconds,
            "official_solved": bool(ok),
            "official_attempt_observed": bool(attempt),
            "previously_solved": bool(prior),
        })

    next_id = (
        rows[-1][0].id
        if has_more and rows
        else None
    )

    return {
        "schema_version": 1,
        "events": events,
        "next_after_id": next_id,
    }, 200, {"Cache-Control": "private, no-store"}
