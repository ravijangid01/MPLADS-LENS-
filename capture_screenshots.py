import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright

SCREENSHOTS_DIR = Path("docs/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("00_Landing", "http://localhost:8501/"),
    ("01_Overview", "http://localhost:8501/Overview"),
    ("02_Flagged_Works", "http://localhost:8501/Flagged_Works"),
    ("03_Work_Detail", "http://localhost:8501/Work_Detail"),
    ("04_Agencies_Districts", "http://localhost:8501/Agencies_Districts"),
    ("05_Data", "http://localhost:8501/Data"),
    ("06_Method_and_Limits", "http://localhost:8501/Method_and_Limits"),
]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        errors = []
        page.on("console", lambda msg: errors.append(f"Console {msg.type}: {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda err: errors.append(f"PageError: {err.message}"))

        for name, url in PAGES:
            print(f"Visiting {name}: {url}")
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(4)  # Allow Streamlit components & plots to finish rendering
            
            # Check for Streamlit runtime exception boxes
            exception_boxes = await page.query_selector_all(".stException")
            if exception_boxes:
                for box in exception_boxes:
                    text = await box.inner_text()
                    errors.append(f"Streamlit Exception on {name}: {text}")

            screenshot_path = SCREENSHOTS_DIR / f"{name}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"Saved screenshot: {screenshot_path}")

        await browser.close()
        
        if errors:
            print("Errors encountered:")
            for e in errors:
                print("  -", e)
        else:
            print("ALL PAGES RENDERED CLEANLY WITH 0 ERRORS!")

if __name__ == "__main__":
    asyncio.run(main())
