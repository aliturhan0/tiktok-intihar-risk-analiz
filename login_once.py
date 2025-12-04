from playwright.sync_api import sync_playwright

PROFILE_DIR = "/Users/aliturhan/tiktok_profile"

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        user_data_dir=PROFILE_DIR,
        headless=False,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-infobars",
        ]
    )
    page = ctx.new_page()
    page.set_default_timeout(300000)
    page.goto("https://www.tiktok.com/", wait_until="load", timeout=300000)
    input("🔑 TikTok’a giriş yap. Giriş bittiyse buraya dönüp Enter’a bas...")
    print("✅ Oturum kaydedildi. Artık scraper çalıştırabilirsin.")
    ctx.close()
