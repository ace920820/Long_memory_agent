// 全局变量
let conversations = new Map();

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    createNewChat();

    // 文本框事件监听
    const textarea = document.getElementById('userInput');
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 初始化移动端菜单
    initMobileMenu();

    // 初始化marked
    marked.setOptions({
        renderer: new marked.Renderer(),
        gfm: true,
        breaks: true,
    });
});

// 初始化移动端菜单
function initMobileMenu() {
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');
    const container = document.querySelector('.container');

    // 创建遮罩层
    const overlay = document.createElement('div');
    overlay.className = 'sidebar-overlay';
    container.appendChild(overlay);

    // 切换菜单状态
    function toggleMenu() {
        sidebar.classList.toggle('active');
        overlay.classList.toggle('active');
    }

    // 点击汉堡菜单按钮
    menuToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleMenu();
    });

    // 点击遮罩层关闭菜单
    overlay.addEventListener('click', toggleMenu);

    // 点击侧边栏内部阻止冒泡
    sidebar.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    // 点击对话项后关闭菜单
    const conversationList = document.getElementById('conversationList');
    conversationList.addEventListener('click', () => {
        toggleMenu();
    });
}

// 创建新对话
function createNewChat() {
    const id = Date.now().toString(); // 仅用于前端标识
    const title = '新对话';
    conversations.set(id, {
        title,
        messages: [],
        serverConvoId: '', // 存储服务器返回的对话ID
        isFirstMessage: true // 标记是否是第一条消息
    });

    const conversationList = document.getElementById('conversationList');
    const item = document.createElement('div');
    item.className = 'conversation-item';
    item.dataset.id = id;
    item.innerHTML = `
        <span>${title}</span>
        <button class="delete-chat" onclick="deleteChat('${id}')">
            <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
        </button>
    `;

    item.addEventListener('click', (e) => {
        if (!e.target.closest('.delete-chat')) {
            switchConversation(id);
        }
    });

    conversationList.appendChild(item);
    switchConversation(id);
}

// 切换对话
function switchConversation(id) {
    // 更新UI激活状态
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.toggle('active', item.dataset.id === id);
    });

    // 清空并重新加载消息
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = '';

    const conversation = conversations.get(id);
    if (conversation) {
        conversation.messages.forEach(msg => {
            appendMessage(msg.role === 'user', msg.content);
        });
    }
}

// 删除对话
function deleteChat(id) {
    conversations.delete(id);
    document.querySelector(`.conversation-item[data-id="${id}"]`).remove();

    // 检查是否还有其他对话
    if (conversations.size === 0) {
        // 如果没有对话了，创建一个新对话
        createNewChat();
    } else if (document.querySelector('.conversation-item.active')?.dataset.id === id) {
        // 如果删除的是当前活动对话，切换到第一个可用对话
        const firstConversation = conversations.keys().next().value;
        switchConversation(firstConversation);
    }
}

// 添加消息到界面
function appendMessage(isUser, content, showLoading = false) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? 'U' : 'A';

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    contentDiv.innerHTML = isUser ? content : marked.parse(content);

    const loadingDots = document.createElement('div');
    loadingDots.className = 'loading-dots';
    loadingDots.innerHTML = `
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
    `;
    loadingDots.style.display = showLoading ? 'flex' : 'none';

    messageContent.appendChild(contentDiv);
    messageContent.appendChild(loadingDots);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    chatContainer.appendChild(messageDiv);

    // 滚动到底部
    chatContainer.scrollTop = chatContainer.scrollHeight;

    return {messageDiv, contentDiv, loadingDots};
}

// 获取当前活动对话
function getCurrentConversation() {
    const activeItem = document.querySelector('.conversation-item.active');
    return activeItem ? conversations.get(activeItem.dataset.id) : null;
}

// 处理SSE数据
function processSSEData(data) {
    try {
        const jsonData = JSON.parse(data);
        if (jsonData.conversation_id) {
            return {type: 'id', content: jsonData.conversation_id};
        } else if (jsonData.content === '[DONE]') {
            return {type: 'done'};
        } else if (jsonData.content) {
            return {type: 'content', content: jsonData.content};
        }
    } catch (e) {
        console.error('解析SSE数据失败:', e);
    }
    return null;
}

// 发送消息
async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();

    if (!message) return;

    // 清空输入框
    input.value = '';
    input.style.height = 'auto';

    const currentConversation = getCurrentConversation();
    if (!currentConversation) return;

    // 显示用户消息
    appendMessage(true, message);
    currentConversation.messages.push({
        role: 'user',
        content: message
    });

    // 创建带加载动画的机器人消息
    const {contentDiv, loadingDots} = appendMessage(false, '', true);
    let responseText = '';

    try {
        // 使用EventSource处理SSE
        const eventSource = new EventSource(`/chatstream?query=${encodeURIComponent(message)}&convo_id=${currentConversation.serverConvoId}`);

        eventSource.onmessage = (event) => {
            const result = processSSEData(event.data);
            if (!result) return;

            switch (result.type) {
                case 'id':
                    currentConversation.serverConvoId = result.content;
                    break;
                case 'content':
                    responseText += result.content;
                    contentDiv.innerHTML = marked.parse(responseText);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    break;
                case 'done':
                    console.log(responseText)
                    loadingDots.style.display = 'none';
                    eventSource.close();
                    // 保存消息到对话历史
                    if (responseText) {
                        currentConversation.messages.push({
                            role: 'assistant',
                            content: responseText
                        });
                    }

                    // 如果是第一条消息，更新对话标题
                    if (currentConversation.isFirstMessage) {
                        currentConversation.isFirstMessage = false;
                        // 获取消息的前10个字符作为标题
                        const newTitle = message.length > 10 ? message.substring(0, 10) + '...' : message;
                        currentConversation.title = newTitle;

                        // 更新左侧对话栏的标题
                        const activeConversationItem = document.querySelector('.conversation-item.active');
                        if (activeConversationItem) {
                            const titleSpan = activeConversationItem.querySelector('span');
                            if (titleSpan) {
                                titleSpan.textContent = newTitle;
                            }
                        }
                    }
                    break;
            }
        };

        eventSource.onerror = (error) => {
            console.error('SSE错误:', error);
            eventSource.close();
            loadingDots.style.display = 'none';
            if (!responseText) {
                responseText = '发生错误: 连接中断';
                contentDiv.textContent = responseText;
            }
            // 保存消息到对话历史
            currentConversation.messages.push({
                role: 'assistant',
                content: responseText
            });
        };

    } catch (error) {
        loadingDots.style.display = 'none';
        responseText = '发生错误: ' + error.message;
        contentDiv.textContent = responseText;
        // 保存消息到对话历史
        currentConversation.messages.push({
            role: 'assistant',
            content: responseText
        });
    }
}
