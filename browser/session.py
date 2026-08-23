from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from camoufox.async_api import AsyncCamoufox
from playwright.async_api import Page

from config import settings

logger = logging.getLogger(__name__)

LOGIN_URL = "https://accounts.google.com/ServiceLogin?service=youtube"
LOGIN_CHECK_URL = "https://www.youtube.com"
SIGNED_IN_SELECTOR = "button#avatar-btn"


def _profile_dir() -> Path:
    settings.ensure_dirs()
    return settings.camoufox_profile_dir


def has_existing_profile() -> bool:
    """Whether a persistent profile already exists on disk."""
    profile = _profile_dir()
    return profile.exists() and any(profile.iterdir())


@asynccontextmanager
async def camoufox_session(
    *, headless: bool | None = None
) -> AsyncIterator[Page]:
    profile = _profile_dir()
    run_headless = settings.camoufox_headless if headless is None else headless

    logger.info("Launching Camoufox (headless=%s, profile=%s)", run_headless, profile)

    async with AsyncCamoufox(
        headless=run_headless,
        persistent_context=True,
        user_data_dir=str(profile),
        humanize=True,
    ) as browser:
        page = await browser.new_page()
        try:
            yield page
        finally:
            try:
                await page.close()
            except Exception:
                pass


async def is_logged_in(page: Page, *, timeout_ms: int = 8000) -> bool:
    await page.goto(LOGIN_CHECK_URL, wait_until="domcontentloaded")
    try:
        await page.wait_for_selector(SIGNED_IN_SELECTOR, timeout=timeout_ms)
        return True
    except Exception:
        return False


async def run_interactive_login() -> None:
    print("Opening a visible browser window for one-time login...")
    print("Please sign in to your Google account.")
    print("This window will close automatically once sign-in is detected.\n")

    async with camoufox_session(headless=False) as page:
        await page.goto(LOGIN_URL, wait_until="domcontentloaded")

        print("Waiting for sign-in to complete...")

        signed_in = False
        browser_closed = False

        for _ in range(180):  # up to ~15 minutes, checked every 5s
            await asyncio.sleep(5)

            try:
                current_url = page.url
            except Exception:
                browser_closed = True
                break

            if "youtube.com" in current_url and "accounts.google.com" not in current_url:
                try:
                    await page.wait_for_selector(
                        SIGNED_IN_SELECTOR, timeout=8000
                    )
                    signed_in = True
                    break
                except Exception:
                    continue

        if browser_closed:
            print(
                "\nThe browser window was closed before sign-in completed. "
                "Re-run 'python -m browser.session --login' to try again, "
                "and leave the window open until it closes itself."
            )
        elif signed_in:
            print("\nSign-in detected. Session saved to:", _profile_dir())
        else:
            print(
                "\nTimed out waiting for sign-in. Re-run 'python -m browser.session "
                "--login' to try again."
            )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Camoufox session management.")
    parser.add_argument(
        "--login",
        action="store_true",
        help="Run one-time interactive login and save the persistent profile.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether the stored profile is currently signed in.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.login:
        asyncio.run(run_interactive_login())
    elif args.check:
        async def _check():
            async with camoufox_session(headless=True) as page:
                ok = await is_logged_in(page)
                print("Signed in." if ok else "Not signed in.")
        asyncio.run(_check())
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
