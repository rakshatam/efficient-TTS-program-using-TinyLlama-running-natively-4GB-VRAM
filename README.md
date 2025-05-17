# efficient-TTS-program-using-TinyLlama-running-natively-4GB-VRAM
worked on the efficient TTS program using TinyLlama running natively (4GB VRAM) and transcribe via whisper library, optimized the LLM response within 3 seconds and can be interrupted mid response by spacebar and take another query
This project implements a lightweight, real-time voice assistant that leverages a local Large Language Model (LLM) running entirely on CPU, combined with robust Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities. The assistant features a user-friendly Gradio interface and supports real-time interruption via a hotkey.

---

## Features

* **Real-time Interaction:** Seamless voice conversation flow.
* **Speech-to-Text (STT):** Powered by OpenAI's Whisper model for accurate transcription of user speech.
* **Large Language Model (LLM):** Uses TinyLlama, a compact yet powerful LLM, for generating responses. Crucially, the LLM inference runs **entirely on CPU** without requiring a dedicated GPU.
* **Text-to-Speech (TTS):** Responses are converted back into speech using `pyttsx3`.
* **Hotkey Interruption:** Interrupt ongoing TTS playback instantly with a customizable hotkey (default: `spacebar`).
* **Gradio Web UI:** Provides an intuitive browser-based interface for easy interaction.
* **Efficient Design:** Utilizes threading for concurrent STT, LLM, and TTS operations, maintaining responsiveness.

---

## How It Works (Pipeline)

The voice assistant operates through a series of concurrent threads and models:

1.  ### **Speech-to-Text (STT) - `STTThread`**
    * Continuously records short audio chunks (1-second duration by default) from the microphone.
    * Temporarily saves each chunk to a WAV file.
    * Transcribes the audio chunk into text using the specified **Whisper model**.
    * Puts the transcribed text into a `transcript_queue` for the LLM.

2.  ### **Large Language Model (LLM) Inference - `LLMThread`**
    * Pulls transcribed user text from the `transcript_queue`.
    * Formats the text into a prompt (`User: [text]\nAssistant:`).
    * Feeds the prompt to the **TinyLlama model** via a Hugging Face `pipeline` configured to run on **CPU (`device=-1`)** using full-precision (`torch.float32`) for compatibility.
    * Extracts the generated assistant response.
    * Puts the generated response text into a `response_queue` for the TTS module.

3.  ### **Text-to-Speech (TTS) - `TTSThread`**
    * Pulls the assistant's text response from the `response_queue`.
    * Converts the text into speech using `pyttsx3`.
    * Plays the generated speech audio.

4.  ### **Hotkey Interruption - `hotkey_listener`**
    * A separate thread listens for a predefined hotkey press (default: `space`).
    * If the hotkey is pressed, it sets an `interrupt_event`.
    * The `TTSThread` continuously checks this event and, if set, stops the current speech playback immediately and clears the event.

5.  ### **Gradio Web UI (`gradio_chat` function)**
    * Handles audio input from the Gradio microphone widget.
    * Transcribes the incoming audio using Whisper.
    * Sends the transcription to the LLM (via `transcript_queue`) and waits for a response (from `response_queue`).
    * Converts the LLM's text response into audio for playback in the Gradio interface.
    * The Gradio interface also provides a visual chat log.

---

## Setup

### Prerequisites

* Python 3.8+
* A microphone
* Local storage for the TinyLlama model and temporary audio files.
