from playwright.sync_api import sync_playwright, TimeoutError
from ocr import read_invoice_5digits

def in_run_window() -> bool:
    # Option simple: exécuter tout le temps (tu peux ajouter une fenêtre horaire plus tard)
    return True

def run():
    if not in_run_window():
        print("Outside run window, exit.")
        return

        target_url = os.getenv("TARGET_URL")
        if not target_url:
            raise RuntimeError("Missing TARGET_URL (set it in GitHub Secrets)")

        
        username = os.getenv("USERNAME")
        password = os.getenv("PASSWORD")
        if not username or not password:
            raise RuntimeError("Missing USERNAME/PASSWORD env vars")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.goto(target_url, wait_until="domcontentloaded")
        page.get_by_placeholder("שם משתמש").fill(username)
        page.get_by_placeholder("סיסמא").fill(password)

        with page.expect_navigation():
            page.get_by_role("button", name="אישור").click()

        page.get_by_role("link", name="הקפץ חדרים פנויים").click()

        title_input = page.locator("#ctl00_ContentPlaceHolder3_txtCaptcha")
        img = page.locator("#ctl00_ContentPlaceHolder3_pnlCapcha img").first
        update_btn = page.get_by_role("button", name="עדכן")

        MAX_RETRIES = 3

        for attempt in range(1, MAX_RETRIES + 1):
            # 1) attend que champ + image soient là
            title_input.wait_for(state="visible", timeout=20000)
            img.wait_for(state="visible", timeout=20000)

            # 2) screenshot + OCR
            img.screenshot(path="debug_image.png")
            invoice = read_invoice_5digits("debug_image.png", debug=True)
            print("invoice number is:", invoice)

            if not invoice:
                print("OCR failed, reload and retry...")
                page.reload(wait_until="domcontentloaded")
                continue

            # 3) IMPORTANT: effacer l'ancien contenu (sinon il reste après refresh)
            title_input.click()
            title_input.press("Control+A")
            title_input.press("Backspace")

            # 4) taper comme un humain (déclenche les events du site)
            title_input.type(invoice, delay=30)

            # 5) pause 2s puis ENTER (comme tu veux)
            page.wait_for_timeout(2000)
            page.keyboard.press("Enter")

            # 6) succès = bouton עדכן visible
            try:
                update_btn.wait_for(state="visible", timeout=8000)
                print("Next page detected ✅ (עדכן visible)")
                break
            except TimeoutError:
                print(f"Not moved to next page (attempt {attempt}). Refresh and retry...")
                page.reload(wait_until="domcontentloaded")

        else:
            raise RuntimeError("Failed to reach next page (עדכן never became visible)")

        # 7) Maintenant clique sur עדכן
        update_btn.click()

        input("Enter pour fermer...")
        context.close()
        browser.close()

if __name__ == "__main__":
    run()