import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  max-width: 600px;
  margin: auto;
  height: 90vh;
  overflow-y: auto;
`;

const BotMessage = styled.div`
  align-self: flex-end;
  background-color: ${props => props.type === "thought" ? "#F5F5F5" :
    props.type === "function_call" ? "#D0D0D0" :
    "#E3FF00"};
  color: black;
  border-radius: 5px;
  padding: 10px;
  margin: 10px;
  white-space: pre-wrap;
`;

const UserMessage = styled.div`
  align-self: flex-start;
  background-color: #66697B;
  color: white;
  border-radius: 5px;
  padding: 10px;
  margin: 10px;
  white-space: pre-wrap;
`;


function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setLoading] = useState(false);
  const bottomRef = useRef();

  const sendMessage = async () => {
    if (!message.trim()) return;  // prevent sending empty message
    setLoading(true);
    const userMessage = { text: message, sender: 'user' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    const response = await fetch(`http://localhost:8000/chat?user_message=${message}`);
    const reader = response.body.getReader();
    const textDecoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const messages = textDecoder.decode(value).split('\n\n');
      for (let msg of messages) {
        if (msg.startsWith('data: ')) {
          const jsonPart = msg.replace('data: ', '');
          const botMessage = JSON.parse(jsonPart);
          botMessage.sender = 'bot';
          setMessages((prevMessages) => [...prevMessages, botMessage]);
        }
      }
    }

    setMessage("");
    setLoading(false);
  };

  const scrollToBottom = () => {
    bottomRef.current.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  return (
    <ChatContainer>
      {messages.map((message, index) => message.sender === 'user' ?
        <UserMessage key={index}>{message.text}</UserMessage> :
        <BotMessage key={index} type={message.type}>{message.text}</BotMessage>
      )}
      <div ref={bottomRef} />
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        disabled={isLoading}
      />
      <button onClick={sendMessage} disabled={isLoading}>Send</button>
    </ChatContainer>
  );
}

export default App;
