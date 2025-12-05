# Project - Story Generator

# imports
import os
import json
import threading
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

# Initialization
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()

# Test print to verify output is working
print("=" * 50, flush=True)
print("Story Generator Starting...", flush=True)
print("=" * 50, flush=True)

# System message for story generation
system_message = "You are a creative story generator assistant. "
system_message += "Generate engaging, age-appropriate stories based on user requests. "
system_message += "Make stories imaginative and fun, with clear characters and plot. "
system_message += "Keep stories concise but complete."

# Story templates and genres
story_genres = {
    "adventure": "An exciting adventure story with heroes, quests, and challenges",
    "fantasy": "A magical fantasy story with wizards, dragons, and enchanted lands",
    "mystery": "A mysterious story with clues, puzzles, and secrets to uncover",
    "sci-fi": "A science fiction story set in the future with technology and space",
    "fairy tale": "A classic fairy tale with magical creatures and happy endings",
    "animal": "A heartwarming story featuring animals as main characters"
}

def get_story_template(genre):
    """Get story template for a specific genre"""
    print(f"Tool get_story_template called for {genre}")
    genre_lower = genre.lower()
    return story_genres.get(genre_lower, "A creative and engaging story")

# Tool definition for story genre selection
story_function = {
    "name": "get_story_template",
    "description": "Get a story template or genre description. Call this when the user wants to generate a story of a specific genre or type.",
    "parameters": {
        "type": "object",
        "properties": {
            "genre": {
                "type": "string",
                "description": "The genre or type of story the user wants (e.g., adventure, fantasy, mystery, sci-fi, fairy tale, animal)",
            },
        },
        "required": ["genre"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": story_function}]

# Handle tool calls
def handle_tool_call(message):
    """Process tool calls from the LLM"""
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    genre = arguments.get('genre')
    template = get_story_template(genre)
    response = {
        "role": "tool",
        "content": json.dumps({"genre": genre, "template": template}),
        "tool_call_id": tool_call.id
    }
    return response, genre

# Image generation for story illustrations
def artist(story_theme):
    """Generate an image illustration for the story"""
    image_response = openai.images.generate(
        model="dall-e-3",
        prompt=f"An illustration for a children's story about {story_theme}, colorful, whimsical, storybook style, suitable for kids",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

# Audio generation (optional - comment out if you have issues)
# Global variable to track audio thread for stopping
_audio_thread = None
_audio_stop_flag = None

try:
    from pydub import AudioSegment
    from pydub.playback import play
    
    def talker(message):
        """Generate and play audio narration of the story in background - starts immediately"""
        global _audio_thread, _audio_stop_flag
        
        # Stop any currently playing audio by setting the flag
        if _audio_stop_flag is not None:
            _audio_stop_flag.set()
        
        # Wait briefly for previous audio to stop (but don't block too long)
        if _audio_thread and _audio_thread.is_alive():
            _audio_thread.join(timeout=0.3)
        
        # Create NEW stop flag for this audio (important - fresh flag for each audio)
        _audio_stop_flag = threading.Event()
        
        # Store reference to this flag in local scope for the thread
        current_stop_flag = _audio_stop_flag
        
        def play_audio():
            try:
                print(f"[Audio Thread] Starting audio generation for message length: {len(message)}", flush=True)
                # Generate audio (this happens in background, doesn't block text display)
                print("[Audio Thread] Calling OpenAI TTS API...", flush=True)
                response = openai.audio.speech.create(
                    model="tts-1",
                    voice="onyx",  # Also try: alloy, echo, fable, onyx, nova, shimmer
                    input=message
                )
                print(f"[Audio Thread] Audio generated successfully, length: {len(response.content)} bytes", flush=True)
                
                # Check if we should stop before playing (use local reference)
                if current_stop_flag.is_set():
                    print("[Audio Thread] Stop flag set, skipping playback", flush=True)
                    return
                
                # Process and play audio
                print("[Audio Thread] Processing audio file...", flush=True)
                audio_stream = BytesIO(response.content)
                audio = AudioSegment.from_file(audio_stream, format="mp3")
                print(f"[Audio Thread] Audio processed, duration: {len(audio)}ms", flush=True)
                
                # Check again before playing (in case stop was called during processing)
                if not current_stop_flag.is_set():
                    print("[Audio Thread] Starting playback NOW...", flush=True)
                    play(audio)
                    print("[Audio Thread] Playback completed", flush=True)
                else:
                    print("[Audio Thread] Stop flag set before playback, skipping", flush=True)
            except Exception as e:
                import traceback
                print(f"\n[Audio Thread] ERROR during audio generation/playback: {e}", flush=True)
                print(f"[Audio Thread] Traceback: {traceback.format_exc()}\n", flush=True)
        
        # Start audio generation/playback in background thread immediately
        # This doesn't block - text appears right away, audio starts generating
        print(f"[talker] Creating audio thread for message: {message[:50]}...", flush=True)
        _audio_thread = threading.Thread(target=play_audio, daemon=True)
        _audio_thread.start()
        print(f"[talker] Thread started, is_alive: {_audio_thread.is_alive()}", flush=True)
    
    def stop_audio():
        """Stop any currently playing audio"""
        global _audio_stop_flag
        if _audio_stop_flag is not None:
            _audio_stop_flag.set()
        
except ImportError:
    print("Audio libraries not available. Audio narration will be skipped.")
    def talker(message):
        pass
    def stop_audio():
        pass
except Exception as e:
    print(f"Audio setup error: {e}")
    def talker(message):
        pass
    def stop_audio():
        pass


# Main chat function with tools - matches week 2 day 5 style exactly
def chat(message, history):
    """Main chat function that handles story generation with tools"""
    # Convert tuple history to dict format (matching week 2 day 5's type="messages" format)
    messages = [{"role": "system", "content": system_message}]
    
    # Convert history from dict format (Gradio 6.x) to OpenAI messages format
    if history:
        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                # Dict format from Gradio - already correct for OpenAI
                messages.append({"role": item["role"], "content": str(item["content"])})
            elif isinstance(item, tuple) and len(item) == 2:
                # Legacy tuple format - convert to dict for OpenAI
                user_msg, assistant_msg = item
                if user_msg:
                    messages.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": str(assistant_msg)})
    
    # Add the current user message
    if message:
        messages.append({"role": "user", "content": str(message)})
    
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    image = None
    
    if response.choices[0].finish_reason == "tool_calls":
        message_obj = response.choices[0].message
        tool_response, genre = handle_tool_call(message_obj)
        # Convert message_obj to dict format (matching week 2 day 5)
        assistant_msg_dict = {
            "role": "assistant",
            "content": message_obj.content if message_obj.content else "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message_obj.tool_calls
            ]
        }
        messages.append(assistant_msg_dict)
        messages.append(tool_response)
        # Generate image based on genre
        image = artist(genre)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        
    reply = response.choices[0].message.content
    if reply is None:
        reply = "I'm sorry, I couldn't generate a response. Please try again."
    
    # Start audio generation IMMEDIATELY as soon as we have the reply text
    # This starts the API call in background right away, before we process history
    # Comment out or delete the next lines if you'd rather skip Audio for now..
    try:
        print(f"\n[Main] ===== Starting audio for reply (length: {len(reply)}) =====", flush=True)
        print(f"[Main] First 100 chars: {reply[:100]}", flush=True)
        talker(reply)  # Starts audio generation in background thread immediately - API call begins now
        print("[Main] Audio thread started successfully\n", flush=True)
    except Exception as e:
        import traceback
        print(f"\n[Main] ERROR starting audio: {e}", flush=True)
        print(f"[Main] Traceback: {traceback.format_exc()}\n", flush=True)
    
    # Add the complete conversation to history as dict format (Gradio 6.x requirement)
    # Gradio 6.x Chatbot expects dict format with 'role' and 'content' keys
    user_msg = str(message).strip() if message else ""
    assistant_msg = str(reply).strip() if reply else ""
    
    # Add messages as dicts with role and content
    if user_msg:
        history.append({"role": "user", "content": user_msg})
    if assistant_msg:
        history.append({"role": "assistant", "content": assistant_msg})
    
    # Return immediately so text appears right away
    # Audio is already generating in background thread
    return history, image

# Gradio UI
with gr.Blocks() as ui:
    gr.Markdown("# 📚 Story Generator - AI-Powered Story Creation")
    gr.Markdown("Ask me to create a story! Try: 'Create an adventure story about a brave knight' or 'Tell me a fantasy story with dragons'")
    
    with gr.Row():
        chatbot = gr.Chatbot(height=500)
        image_output = gr.Image(height=500, label="Story Illustration")
    
    with gr.Row():
        entry = gr.Textbox(label="What kind of story would you like?", placeholder="e.g., Create a mystery story about a lost treasure...")
    
    with gr.Row():
        clear = gr.Button("Clear Chat")

    def respond(message, history):
        """Handle user input and generate response"""
        if not message or not message.strip():
            return "", history or [], None
        
        # Initialize history if needed
        if history is None:
            history = []
        if not isinstance(history, list):
            history = []
        
        # Clean history - convert tuples to dicts or keep dicts
        clean_history = []
        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                # Already in correct dict format
                clean_history.append({"role": item["role"], "content": str(item["content"])})
            elif isinstance(item, tuple) and len(item) == 2:
                # Convert tuple to dict format
                part1, part2 = item
                if part1:
                    clean_history.append({"role": "user", "content": str(part1)})
                if part2:
                    clean_history.append({"role": "assistant", "content": str(part2)})
        
        # Call chat function
        updated_history, image = chat(message, clean_history)
        
        # Return cleaned history - ensure all are dicts with role and content
        result_history = []
        for item in updated_history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                # Already in correct format
                result_history.append({"role": item["role"], "content": str(item["content"])})
            elif isinstance(item, tuple) and len(item) == 2:
                # Convert tuple to dict format
                part1, part2 = item
                if part1:
                    result_history.append({"role": "user", "content": str(part1)})
                if part2:
                    result_history.append({"role": "assistant", "content": str(part2)})
        
        return "", result_history, image

    def stop_audio_and_clear():
        """Stop audio and clear chat"""
        try:
            stop_audio()
        except:
            pass
        return [], None
    
    entry.submit(respond, inputs=[entry, chatbot], outputs=[entry, chatbot, image_output])
    clear.click(stop_audio_and_clear, inputs=None, outputs=[chatbot, image_output], queue=False)

if __name__ == "__main__":
    ui.launch(inbrowser=True)

