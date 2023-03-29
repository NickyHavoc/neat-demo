<script>
	import axios from "axios";

	let inputMessage = "";
	let messages = [];

	async function sendMessage() {
		if (inputMessage.trim() === "") return;
		
		const messageObj = { message: inputMessage, sender: "user" };
		messages = [...messages, messageObj];

		try {
		const response = await axios.post("http://localhost:8000/send-message", {
			message: inputMessage
		});
		const botMessage = { message: response.data.message, sender: "bot" };
		messages = [...messages, botMessage];
		} catch (error) {
		console.error("Error sending message:", error);
		}

		inputMessage = "";
	}
</script>

<style>
	html, body {
		height: 100%;
		margin: 0;
	}

	.container {
		display: flex;
		flex-direction: column;
		justify-content: flex-end;
		height: 100%;
		max-width: 800px;
		margin: 0 auto;
	}

	.messages {
		width: 100%;
		max-height: 80vh;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
	}

	.message {
		padding: 10px;
		margin: 5px;
		border-radius: 5px;
	}

	.user {
		align-self: flex-end;
		background-color: #0abab5;
		color: white;
	}

	.bot {
		align-self: flex-start;
		background-color: white;
		color: black;
		border: 1px solid #ccc;
	}

	.input-container {
		display: flex;
		width: 100%;
		margin-top: 10px;
	}

	input {
		flex-grow: 1;
		padding: 10px;
		border: 1px solid #ccc;
		border-radius: 5px;
	}

	button {
		padding: 10px;
		margin-left: 5px;
		border: none;
		background-color: #0abab5;
		color: white;
		cursor: pointer;
		border-radius: 5px;
	}

	button:hover {
		background-color: #089a9a;
	}
</style>

<div class="container">
	<div class="messages">
		{#each messages as { message, sender }}
		<div class={`message ${sender}`}>
			{message}
		</div>
		{/each}
	</div>
	<div class="input-container">
		<input
		type="text"
		bind:value="{inputMessage}"
		on:keydown="{e => (e.key === 'Enter' ? sendMessage() : null)}"
		placeholder="Type your message..."
		/>
		<button on:click="{sendMessage}">Send</button>
	</div>
</div>
