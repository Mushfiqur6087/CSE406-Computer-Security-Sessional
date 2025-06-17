import time
import json
import os
import signal
import sys
import random
import traceback
import socket
import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import database
from database import Database

WEBSITES = [
    # websites of your choice
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

TRACES_PER_SITE = 1000
FINGERPRINTING_URL = "http://localhost:5000" 
OUTPUT_PATH = "dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

"""
Helper functions to make your life easier.
"""

def is_server_running(host='127.0.0.1', port=5000):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

async def retrieve_traces_from_backend(page):
    """Retrieve traces from the backend API."""
    try:
        # Try the correct endpoint from our app.py implementation
        traces = await page.evaluate("""
            () => {
                return fetch('/api/get_results')
                    .then(response => response.ok ? response.json() : {traces: []})
                    .then(data => data.traces || [])
                    .catch(() => []);
            }
        """)
        
        count = len(traces) if traces else 0
        print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
        return traces or []
    except Exception as e:
        print(f"  - Error retrieving traces: {e}")
        return []

async def clear_trace_results(page):
    """Clear all results from the backend by pressing the button."""
    try:
        # Find and click the clear button
        await page.click("text=Clear Results")
        
        # Wait for success message
        await page.wait_for_selector("div[role='alert']:has-text('Cleared')", timeout=5000)
        print("  - Successfully cleared results")
    except Exception as e:
        print(f"  - Error clearing results: {e}")
    
def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

"""
Your implementation starts here.
"""

async def collect_single_trace(browser, website_url, website_index):
    """ Implement the trace collection logic here. 
    1. Open the fingerprinting website
    2. Click the button to collect trace
    3. Open the target website in a new tab
    4. Interact with the target website (scroll, click, etc.)
    5. Return to the fingerprinting tab and close the target website tab
    6. Wait for the trace to be collected
    7. Return success or failure status
    """
    try:
        # Create a new browser context for isolation
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        # 1. Open the fingerprinting website  
        fingerprint_page = await context.new_page()
        await fingerprint_page.goto(FINGERPRINTING_URL, wait_until='domcontentloaded')
        print(f"  - Opened fingerprinting page: {FINGERPRINTING_URL}")
        
        # Debug: Check what's on the page
        try:
            title = await fingerprint_page.title()
            print(f"  - Page title: {title}")
            
            # Check if our expected elements exist
            buttons = await fingerprint_page.query_selector_all("button")
            print(f"  - Found {len(buttons)} buttons on page")
            
        except Exception as e:
            print(f"  - Debug info error: {e}")
        
        # 2. Click the button to collect trace
        try:
            # Wait for the page to load completely
            await fingerprint_page.wait_for_load_state('networkidle', timeout=10000)
            
            # Try different selectors for the collect trace button
            button_selectors = [
                "text=Collect Trace Data",
                "button:has-text('Collect Trace Data')",
                "#collectBtn",
                "button[onclick*='collectTraceData']"
            ]
            
            clicked = False
            for selector in button_selectors:
                try:
                    await fingerprint_page.click(selector, timeout=5000)
                    clicked = True
                    print(f"  - Started trace collection using selector: {selector}")
                    break
                except:
                    continue
            
            if not clicked:
                print(f"  - Warning: Could not find collect trace button")
                # Try clicking any button that looks like it could be the trace collection
                await fingerprint_page.click("button", timeout=5000)
                print(f"  - Clicked first available button as fallback")
                
        except Exception as e:
            print(f"  - Error clicking collect trace button: {e}")
            await context.close()
            return False
        
        # 3. Open the target website in a new tab
        target_page = await context.new_page()
        await target_page.goto(website_url, wait_until='domcontentloaded', timeout=30000)
        print(f"  - Opened target website: {website_url}")
        
        # 4. Interact with the target website (scroll, click, etc.)
        await simulate_user_activity(target_page)
        
        # 5. Return to the fingerprinting tab and close the target website tab
        await target_page.close()
        await fingerprint_page.bring_to_front()
        
        # 6. Wait for the trace to be collected (10 seconds as per specification)
        print(f"  - Waiting for trace collection to complete...")
        
        # Wait for the trace collection to start and complete
        await asyncio.sleep(2)  # Wait for collection to start
        
        # Check if trace collection is in progress by looking for UI indicators
        try:
            # Wait for the collection to finish - look for success message or completed state
            await asyncio.sleep(12)  # Wait for full collection time (10s + buffer)
            print(f"  - Trace collection time completed, retrieving data...")
        except Exception as e:
            print(f"  - Warning during trace collection wait: {e}")
        
        # 7. Retrieve the trace data - try multiple times if needed
        traces = None
        for attempt in range(3):  # Try up to 3 times
            print(f"  - Attempt {attempt + 1}/3 to retrieve traces...")
            traces = await retrieve_traces_from_backend(fingerprint_page)
            if traces:
                break
            await asyncio.sleep(2)  # Wait between attempts
        
        if traces:
            # Save the most recent trace to database
            latest_trace = traces[-1] if traces else None
            if latest_trace:
                success = database.db.save_trace(website_url, website_index, latest_trace)
                if success:
                    print(f"  - Successfully saved trace for {website_url}")
                    await clear_trace_results(fingerprint_page)
                    await context.close()
                    return True
        
        print(f"  - Failed to collect trace for {website_url}")
        await context.close()
        return False
        
    except Exception as e:
        print(f"  - Error collecting trace for {website_url}: {e}")
        try:
            await context.close()
        except:
            pass
        return False

async def simulate_user_activity(page):
    """Simulate realistic user activity on the target website."""
    try:
        # Get page dimensions
        viewport = await page.evaluate("() => ({width: window.innerWidth, height: window.innerHeight})")
        
        # Scroll randomly
        for _ in range(random.randint(3, 8)):
            scroll_y = random.randint(0, 1000)
            await page.evaluate(f"window.scrollTo(0, {scroll_y})")
            await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Try to click on some elements (safely)
        try:
            # Look for common clickable elements
            clickable_selectors = [
                "a[href]", "button", "input[type='button']", 
                "input[type='submit']", "[onclick]", ".btn"
            ]
            
            for selector in clickable_selectors[:2]:  # Try first 2 selectors
                elements = await page.query_selector_all(selector)
                if elements and len(elements) > 0:
                    # Click a random element
                    element = random.choice(elements[:5])  # Limit to first 5 elements
                    await element.click()
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                    break
        except:
            pass  # Continue even if clicking fails
            
        # Final scroll
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(0.5)
        
    except Exception as e:
        print(f"    - Warning: Error during user simulation: {e}")

async def collect_fingerprints(browser, target_counts=None):
    """ Implement the main logic to collect fingerprints.
    1. Calculate the number of traces remaining for each website
    2. Collect traces for each website until the target number is reached
    3. Save the traces to the database
    4. Return the total number of new traces collected
    """
    
    if target_counts is None:
        # 1. Calculate the number of traces remaining for each website
        current_counts = database.db.get_traces_collected()
        target_counts = {website: max(0, TRACES_PER_SITE - current_counts.get(website, 0)) 
                        for website in WEBSITES}
    
    total_collected = 0
    
    # 2. Collect traces for each website until the target number is reached
    for website_index, website_url in enumerate(WEBSITES):
        needed = target_counts.get(website_url, 0)
        if needed <= 0:
            print(f"✅ Website {website_url}: Already has enough traces ({TRACES_PER_SITE}/{TRACES_PER_SITE})")
            continue
            
        print(f"\n🎯 Website {website_url}: Collecting {needed} traces...")
        print(f"   Progress: [{'█' * (TRACES_PER_SITE - needed)}{'░' * needed}] {TRACES_PER_SITE - needed}/{TRACES_PER_SITE}")
        
        collected_for_site = 0
        for trace_num in range(needed):
            print(f"\n   📊 Collecting trace {trace_num + 1}/{needed} for {website_url}")
            
            success = await collect_single_trace(browser, website_url, website_index)
            
            if success:
                collected_for_site += 1
                total_collected += 1
                print(f"   ✅ Trace {trace_num + 1} collected successfully!")
            else:
                print(f"   ❌ Failed to collect trace {trace_num + 1}")
            
            # Small delay between traces
            await asyncio.sleep(random.uniform(2, 5))
        
        final_count = database.db.get_traces_collected().get(website_url, 0)
        print(f"\n✅ Website {website_url}: Completed! Collected {collected_for_site}/{needed} new traces")
        print(f"   📊 Total for this site: {final_count}/{TRACES_PER_SITE}")
        print("=" * 60)
    
    return total_collected

async def main():
    """ Implement the main function to start the collection process.
    1. Check if the Flask server is running
    2. Initialize the database
    3. Set up the Browser
    4. Start the collection process, continuing until the target number of traces is reached
    5. Handle any exceptions and ensure the browser is closed at the end
    6. Export the collected data to a JSON file
    7. Retry if the collection is not complete
    """
    
    print("🚀 Starting automated data collection with Playwright...")
    print("=" * 60)
    print("📋 COLLECTION PLAN:")
    print(f"   📊 Target: {TRACES_PER_SITE} traces per website")
    print(f"   🌐 Websites: {len(WEBSITES)} total")
    for i, website in enumerate(WEBSITES):
        print(f"      {i+1}. {website}")
    print(f"   🎯 Total target: {len(WEBSITES) * TRACES_PER_SITE} traces")
    print(f"   🤖 Mode: Headless browser automation")
    print("=" * 60)
    
    # 1. Check if the Flask server is running
    if not is_server_running():
        print("❌ Flask server is not running. Please start it with: python app.py")
        return
    
    print("✓ Flask server is running")
    
    # 2. Initialize the database
    try:
        database.db.init_database()
        print("✓ Database initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return
    
    # 3. Set up the Browser
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(
                headless=True,  # Running in headless mode for automation
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding'
                ]
            )
            print("✓ Browser launched successfully")
            
            # 4. Start the collection process
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"\n� Collection attempt {attempt + 1}/{max_retries}")
                    print("=" * 60)
                    
                    # Show current status with progress bars
                    current_counts = database.db.get_traces_collected()
                    total_current = sum(current_counts.values())
                    total_target = len(WEBSITES) * TRACES_PER_SITE
                    
                    print("📊 Current collection status:")
                    for website, count in current_counts.items():
                        progress_bar = '█' * count + '░' * (TRACES_PER_SITE - count)
                        percentage = (count / TRACES_PER_SITE) * 100
                        print(f"   {website}")
                        print(f"   [{progress_bar}] {count}/{TRACES_PER_SITE} ({percentage:.1f}%)")
                    
                    overall_percentage = (total_current / total_target) * 100
                    print(f"\n🎯 Overall Progress: {total_current}/{total_target} ({overall_percentage:.1f}%)")
                    print("=" * 60)
                    
                    if is_collection_complete():
                        print("✅ Collection complete! All targets reached.")
                        break
                    
                    # Collect more traces
                    new_traces = await collect_fingerprints(browser)
                    print(f"📈 Collected {new_traces} new traces in this attempt")
                    
                    if is_collection_complete():
                        print("✅ Collection complete!")
                        break
                        
                except Exception as e:
                    print(f"❌ Error in collection attempt {attempt + 1}: {e}")
                    traceback.print_exc()
                    if attempt < max_retries - 1:
                        print("⏳ Retrying in 10 seconds...")
                        await asyncio.sleep(10)
            
            # 6. Export the collected data to a JSON file
            try:
                database.db.export_to_json(OUTPUT_PATH)
                print(f"✅ Data exported to {OUTPUT_PATH}")
            except Exception as e:
                print(f"❌ Failed to export data: {e}")
            
            # Final status
            final_counts = database.db.get_traces_collected()
            total_final = sum(final_counts.values())
            total_target = len(WEBSITES) * TRACES_PER_SITE
            
            print("\n" + "=" * 60)
            print("🏁 FINAL COLLECTION REPORT")
            print("=" * 60)
            for website, count in final_counts.items():
                progress_bar = '█' * count + '░' * (TRACES_PER_SITE - count)
                percentage = (count / TRACES_PER_SITE) * 100
                status = "✅ COMPLETE" if count >= TRACES_PER_SITE else "⚠️  INCOMPLETE"
                print(f"{status} {website}")
                print(f"         [{progress_bar}] {count}/{TRACES_PER_SITE} ({percentage:.1f}%)")
            
            overall_percentage = (total_final / total_target) * 100
            print(f"\n🎯 OVERALL: {total_final}/{total_target} traces ({overall_percentage:.1f}%)")
            
            if total_final >= total_target:
                print("🎉 COLLECTION SUCCESSFUL! All targets reached.")
            else:
                print("⚠️  COLLECTION INCOMPLETE. Some targets not reached.")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Critical error: {e}")
            traceback.print_exc()
        finally:
            # 5. Ensure the browser is closed at the end
            try:
                await browser.close()
                print("✓ Browser closed")
            except:
                pass

def run_main():
    """Wrapper to run the async main function."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_main()
