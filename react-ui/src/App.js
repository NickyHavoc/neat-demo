import React, { useState, useEffect, useRef } from 'react';
import styled, { createGlobalStyle } from 'styled-components';

const GlobalStyle = createGlobalStyle`
  body {
    font-family: 'Verdana', sans-serif;
    background-color: #232528;
    font-size: 12px;
  }
`;

const InputBar = styled.input`
  padding: 20px;
  border-radius: 5px;
  border: none;
  width: 80%;
  box-sizing: border-box;
  outline: none;
  font-size: 12px;
`;

const SendButton = styled.button`
  padding: 20px;
  border-radius: 5px;
  border: none;
  background-color: #66697B;
  color: white;
  width: 15%;
  cursor: pointer;
  outline: none;
  font-size: 12px;
  &:disabled {
    background: #ddd;
    color: #666;
  }
`;

const MessagesWrapper = styled.div`
  overflow-y: auto;
`;

const Form = styled.form`
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  bottom: 0;
  background-color: #232528;
  padding: 20px 0;
`;

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  max-width: 70vw;
  margin: auto;
  height: 90vh;
`;

const MessageContainer = styled.div`
  display: flex;
  justify-content: ${props => props.sender === 'user' ? "flex-end" : "flex-start"};
  width: 95%;
  padding: 10px;
`;

const Message = styled.div`
  max-width: 80%;
  background-color: ${props => props.type === "thought" ? "#F5F5F5" :
    props.type === "function_call" ? "#D0D0D0" :
    props.sender === 'user' ? "#66697B" : "#E3FF00"};
  color: ${props => props.sender === 'user' ? "white" : "black"};
  border-radius: 5px;
  padding: 20px;
  white-space: pre-wrap;
  overflow-wrap: break-word;
`;

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setLoading] = useState(false);
  const bottomRef = useRef();

  const sendMessage = async (e) => {
    e.preventDefault(); 
    if (!message.trim()) return;
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
      messages
        .filter(msg => msg.startsWith('data: '))
        .map(msg => msg.replace('data: ', ''))
        .forEach(jsonPart => {
          const botMessage = { ...JSON.parse(jsonPart), sender: 'bot' };
          setMessages((prevMessages) => [...prevMessages, botMessage]);
        });
    }

    setMessage("");
    setLoading(false);
  };

  const scrollToBottom = () => {
    bottomRef.current.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);


  return (
    <>
      <GlobalStyle />
      <ChatContainer>
        <MessagesWrapper>
          {messages.map((message, index) =>
            <MessageContainer key={index} sender={message.sender}>
              <Message {...message}>{message.text}</Message>
            </MessageContainer>
          )}
          <div ref={bottomRef} />
        </MessagesWrapper>
        <Form onSubmit={sendMessage}>
          <InputBar
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            disabled={isLoading}
          />
          <SendButton type="submit" disabled={isLoading}>Send</SendButton>
        </Form>
      </ChatContainer>
    </>
  );
}

export default App;
