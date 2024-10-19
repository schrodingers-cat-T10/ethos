import React, { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

const Chatbot = () => {
  const [selectedModel, setSelectedModel] = useState("Llama 3.1");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null); // Reference to the file input
  const recognitionRef = useRef(null); // Reference to Speech Recognition API

  // Initialize SpeechRecognition if available
  if (!recognitionRef.current && window.SpeechRecognition) {
    recognitionRef.current = new window.SpeechRecognition();
  }

  const handleSelectChange = (event) => {
    setSelectedModel(event.target.value);
  };

  // Handle file input click (simulating click for file input field)
  const handleFileClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file); // Update state with selected file
  };

  // Handle drag-and-drop events
  const handleDragOver = (event) => {
    event.preventDefault(); // Prevent default behavior (like opening the file)
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    setSelectedFile(file); // Set the dropped file as selected file
  };

  // Function to handle TTS
  const handleTextToSpeech = (text) => {
    const speech = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(speech);
  };

  // Function to handle speech-to-text (STT)
  const handleSpeechToText = () => {
    if (recognitionRef.current) {
      recognitionRef.current.start();

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript); // Set the recognized speech to the input field
      };

      recognitionRef.current.onerror = (event) => {
        console.error("Error with Speech Recognition:", event.error);
      };
    }
  };

  const handleSendMessage = async () => {
    if (input.trim() || selectedFile) {
      // Add the user's message to the chat
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "user", text: input },
      ]);
      const userMessage = input;
      setInput("");

      // Prepare form data for file upload
      const formData = new FormData();
      formData.append("input_text", userMessage); // Add user message
      if (selectedFile) {
        formData.append("file", selectedFile); // Add the selected file
      }

      try {
        // Make a POST request to the FastAPI backend
        const response = await axios.post(
          "http://localhost:8000/ask/",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data", // Specify form-data for file upload
            },
          }
        );

        // Add the bot's response to the chat
        const botResponse = response.data.response.output;
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: selectedModel, text: botResponse },
        ]);

        // Call TTS to speak the bot's response
        handleTextToSpeech(botResponse);
      } catch (error) {
        console.error("Error communicating with the backend:", error);
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: "bot", text: "Error communicating with the backend." },
        ]);
      }

      // Clear the file input after sending
      setSelectedFile(null);
    }
  };

  return (
    <div className="chatbot-container">
      <div className="chat-header">
        <h3>01INSTRUCTORs</h3>
        <div className="dropdown-container">
          <label htmlFor="model-select">Model:</label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={handleSelectChange}
          >
            <option value="Llama 3.1">Llama 3.1</option>
            <option value="Gemini">Gemini</option>
            <option value="OpenAI">OpenAI</option>
          </select>
        </div>
      </div>
      <div className="chat-window">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`chat-message ${
              message.sender === "user" ? "user-message" : "bot-message"
            }`}
          >
            <strong>{message.sender}: </strong> {message.text}
          </div>
        ))}
      </div>
      <div className="chat-input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
        />

        {/* Microphone button for speech-to-text */}
        <button onClick={handleSpeechToText} className="mic-button">
          🎤
        </button>

        {/* Drag-and-drop file upload area */}
        <div
          className="file-drop-area"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={handleFileClick} // Trigger file input on click
        >
          {selectedFile ? (
            <div className="file-preview">
              <span>{selectedFile.name}</span>
              <button onClick={() => setSelectedFile(null)}>Remove</button>
            </div>
          ) : (
            <span>Drag & drop a file or click to upload</span>
          )}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: "none" }} // Hide the input field
          />
        </div>

        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
};

export default Chatbot;
