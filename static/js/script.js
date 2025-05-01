// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');

    // Function to add messages to the chat
    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // If message is a string, display as text
        if (typeof message === 'string') {
            messageContent.textContent = message;
        } else {
            // If message is an object (diagnosis result)
            if (message.status === 'error') {
                messageContent.textContent = message.message;
            } else {
                // Create HTML content for the diagnosis
                let html = '';
                
                // Display identified symptoms
                if (message.identified_symptoms && message.identified_symptoms.length > 0) {
                    html += '<div class="symptoms-list">';
                    html += '<strong>Identified symptoms:</strong> ';
                    html += message.identified_symptoms.join(', ');
                    html += '</div>';
                }
                
                // Display prediction results
                if (message.predictions && message.predictions.length > 0) {
                    html += '<p><strong>Possible conditions:</strong></p>';
                    
                    message.predictions.forEach(prediction => {
                        html += `<div class="prediction-card">`;
                        html += `<p><strong>${prediction.disease}</strong> (${prediction.probability}%)</p>`;
                        html += `<div class="prediction-probability">`;
                        html += `<div class="prediction-fill" style="width: ${prediction.probability}%"></div>`;
                        html += `</div>`;
                        html += `</div>`;
                    });
                }
                
                // Add disclaimer
                if (message.disclaimer) {
                    html += `<p class="disclaimer">${message.disclaimer}</p>`;
                }
                
                messageContent.innerHTML = html;
            }
        }
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to process user input
    async function processUserInput() {
        const userMessage = userInput.value.trim();
        
        if (userMessage === '') {
            return;
        }
        
        // Add user message to chat
        addMessage(userMessage, true);
        
        // Clear input field
        userInput.value = '';
        
        // Add loading message
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot-message';
        loadingDiv.innerHTML = '<div class="message-content"><p>Analyzing symptoms...</p></div>';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        try {
            // Send request to backend
            const response = await fetch('/diagnose', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symptoms: userMessage }),
            });
            
            const result = await response.json();
            
            // Remove loading message
            chatMessages.removeChild(loadingDiv);
            
            // Display result
            addMessage(result);
            
        } catch (error) {
            // Remove loading message
            chatMessages.removeChild(loadingDiv);
            
            // Display error message
            addMessage('Sorry, there was an error processing your request. Please try again.');
            console.error('Error:', error);
        }
    }

    // Event listeners
    sendBtn.addEventListener('click', processUserInput);
    
    userInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            processUserInput();
        }
    });

    // Focus input on page load
    userInput.focus();
});