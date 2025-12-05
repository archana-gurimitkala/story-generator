# Story Generator

An AI-powered story generator built with OpenAI and Gradio.

## Demo

Watch the demo video to see the story generator in action:

https://github.com/archana-gurimitkala/story-generator/blob/main/demo.mp4

## Features

- 🤖 AI-powered story generation using OpenAI GPT-4o-mini
- 🎨 Automatic story illustration generation with DALL-E 3
- 🔊 Text-to-speech narration (optional)
- 🛠️ Tool/function calling for genre selection
- 💬 Interactive Gradio chat interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. Run the application:
```bash
python story_generator.py
```

## Usage

1. Launch the application - a browser window will open automatically
2. Type your story request, for example:
   - "Create an adventure story about a brave knight"
   - "Tell me a fantasy story with dragons"
   - "Generate a mystery story about a lost treasure"
3. The AI will generate a story and create an illustration
4. The story will also be narrated (if audio is enabled)

## Story Genres

The generator supports various story genres:
- Adventure
- Fantasy
- Mystery
- Sci-Fi
- Fairy Tale
- Animal stories

## Audio Setup (Optional)

If you want audio narration, you may need to install FFmpeg:

**Mac:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

**Linux:**
```bash
sudo apt-get install ffmpeg
```

If you encounter audio issues, you can comment out the `talker()` function call in the code.

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection

## License

Open source project - feel free to use and modify as needed.


