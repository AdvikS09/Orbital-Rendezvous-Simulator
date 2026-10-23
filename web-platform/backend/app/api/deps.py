from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app import bootstrap  # noqa: F401

from pilot.analytics import ParticipantContext  # noqa: E402
from pilot.auth import get_authenticator  # noqa: E402
from pilot.config import get_settings  # noqa: E402
from pilot.db import get_engine  # noqa: E402
from pilot.models import organisations, participants, pilot_codes, sessions  # noqa: E402

from sqlalchemy import select  # noqa: E402

_basic = HTTPBasic(auto_error=False)


def load_context(session_token: str | None) -> ParticipantContext | None:
    if not session_token:
        return None
    try:
        engine = get_engine()
        with engine.connect() as connection:
            row = connection.execute(
                select(
                    sessions.c.id,
                    sessions.c.participant_id,
                    sessions.c.pilot_code_id,
                    sessions.c.organisation_id,
                    sessions.c.track,
                    sessions.c.is_return_visit,
                    participants.c.participant_ref,
                    participants.c.user_type,
                    organisations.c.name.label("organisation_name"),
                    pilot_codes.c.code.label("pilot_code"),
                )
                .select_from(
                    sessions.join(
                        participants, participants.c.id == sessions.c.participant_id
                    )
                    .outerjoin(
                        organisations,
                        organisations.c.id == sessions.c.organisation_id,
                    )
                    .outerjoin(
                        pilot_codes, pilot_codes.c.id == sessions.c.pilot_code_id
                    )
                )
                .where(sessions.c.session_token == session_token)
            ).mappings().first()
    except Exception:  # noqa: BLE001 - analytics must never break a simulation
        return None

    if row is None:
        return None

    return ParticipantContext(
        participant_id=int(row["participant_id"]),
        participant_ref=row["participant_ref"],
        session_id=int(row["id"]),
        session_token=session_token,
        organisation_id=row["organisation_id"],
        pilot_code_id=row["pilot_code_id"],
        user_type=row["user_type"],
        track=row["track"],
        is_return_visit=bool(row["is_return_visit"]),
        organisation_name=row["organisation_name"],
        pilot_code=row["pilot_code"],
    )


def optional_participant(
    x_session_token: Annotated[str | None, Header(alias="X-Session-Token")] = None,
) -> ParticipantContext | None:
    return load_context(x_session_token)


def require_admin(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(_basic)] = None,
) -> str:
    settings = get_settings()
    authenticator = get_authenticator(settings)

    if not authenticator.configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No administrator credentials are configured on this deployment. "
                "Set PILOT_ADMIN_USERNAME and PILOT_ADMIN_PASSWORD_HASH."
            ),
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Administrator credentials required.",
            headers={"WWW-Authenticate": 'Basic realm="Pilot administration"'},
        )

    result = authenticator.authenticate(credentials.username, credentials.password)
    if not result.ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result.message,
            headers={"WWW-Authenticate": 'Basic realm="Pilot administration"'},
        )

    secrets.compare_digest(credentials.username, settings.admin_username)
    return credentials.username
