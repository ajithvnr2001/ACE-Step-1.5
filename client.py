import requests
import json
import time
import os

def generate_and_download(api_url, prompt, lyrics="", vocal_language="en", thinking=False, output_filename="output.mp3"):
    """
    Submits a music generation task to the Modal API, polls for completion, and downloads the result.
    
    Args:
        api_url (str): The base URL of your Modal API (e.g., https://...modal.run)
        prompt (str): Description of the music (e.g., "A soulful jazz track with a smooth saxophone lead")
        lyrics (str): Lyrics for the song (optional)
        vocal_language (str): Language code (e.g., "en", "zh")
        thinking (bool): If True, uses the 5Hz LM for better quality (highly recommended)
        output_filename (str): Where to save the resulting audio file
    """
    api_url = api_url.rstrip("/")
    
    # 1. Submit Task
    print(f"🚀 Submitting task to {api_url}...")
    payload = {
        "prompt": prompt,
        "lyrics": lyrics,
        "vocal_language": vocal_language,
        "thinking": thinking,
        "inference_steps": 8,
        "guidance_scale": 7.0
    }
    
    try:
        response = requests.post(f"{api_url}/release_task", json=payload)
        response.raise_for_status()
        task_data = response.json()
        task_id = task_data["data"]["task_id"]
        print(f"✅ Task created! Task ID: {task_id}")
    except Exception as e:
        print(f"❌ Failed to submit task: {e}")
        return

    # 2. Poll for Result
    print("⏳ Waiting for generation (this may take 1-2 minutes)...")
    result_url = f"{api_url}/query_result"
    
    max_retries = 120  # 10 minutes max (5s * 120)
    for i in range(max_retries):
        try:
            # IMPORTANT: API expects 'task_id_list' as a JSON-encoded string of a list
            query_payload = {"task_id_list": json.dumps([task_id])}
            res = requests.post(result_url, json=query_payload)
            res.raise_for_status()
            
            response_json = res.json()
            if not response_json.get("data"):
                time.sleep(5)
                continue
                
            data = response_json["data"][0]
            status = data["status"]
            
            # 0: queued/running, 1: succeeded, 2: failed
            if status == 1:
                # The 'result' field is a second JSON-encoded string containing a list of result objects
                result_list = json.loads(data["result"])
                if result_list and result_list[0].get("file"):
                    audio_link = result_list[0]["file"]
                    print(f"🎉 Success! Downloading audio from {audio_link}...")
                    
                    # 3. Download Audio
                    audio_res = requests.get(audio_link)
                    audio_res.raise_for_status()
                    with open(output_filename, "wb") as f:
                        f.write(audio_res.content)
                    
                    print(f"💾 File saved successfully as: {os.path.abspath(output_filename)}")
                    return os.path.abspath(output_filename)
                else:
                    print("⚠️ Task succeeded but no audio file was found.")
                    return
            
            elif status == 2:
                # Parse error if available
                result_list = json.loads(data.get("result", "[]"))
                err_msg = result_list[0].get("error") if result_list else "Unknown error"
                print(f"❌ Generation failed: {err_msg}")
                return
            
            # Print status every few seconds
            if i % 4 == 0:
                status_text = "Processing" if status == 0 else "Pending"
                print(f"   [Status]: {status_text}...")
                
        except Exception as e:
            print(f"⚠️ Polling error: {e}")
            
        time.sleep(5)
        
    print("⏰ Timeout: Generation took too long.")

# --- COLAB USAGE EXAMPLE ---
if __name__ == "__main__":
    # 1. Replace with your actual Modal API URL
    MY_API_URL = "https://infotamil000--acestep-api-api-app.modal.run" 
    
    # 2. Set your music parameters
    PROMPT = "Upbeat synthwave with 80s vibes"
    LYRICS = "[instrumental]"
    
    # 3. Run it!
    generate_and_download(
        api_url=MY_API_URL,
        prompt=PROMPT,
        lyrics=LYRICS,
        output_filename="my_generated_music.mp3"
    )
