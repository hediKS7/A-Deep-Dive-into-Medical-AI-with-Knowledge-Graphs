import streamlit as st

class MedicalDarkTheme:
    """
    Ultra-Modern Medical AI Chatbot Theme v2.0
    Enhanced with advanced animations and user experience features
    """

    @staticmethod
    def apply():
        st.markdown("""
        <style>

        /* ================= ROOT ================= */
        :root {
            /* Tanit.ai Brand */
            --tanit-teal: #1AC3AE;
            --tanit-teal-light: #6AE8D8;
            --tanit-purple: #6D529F;
            --tanit-purple-light: #9D7FD3;
            --tanit-indigo: #3F4080;
            
            /* Gradients */
            --gradient-medical: linear-gradient(135deg, 
                var(--tanit-teal) 0%, 
                var(--tanit-purple) 50%,
                var(--tanit-indigo) 100%);
            --gradient-neuro: linear-gradient(45deg,
                #1AC3AE 0%,
                #00B5B8 25%,
                #6D529F 75%,
                #3F4080 100%);

            /* Status Colors */
            --success: #4CAF50;
            --warning: #FF9800;
            --danger: #EF5350;
            --info: #2196F3;

            /* Dark UI */
            --bg-main: #0A0D12;
            --bg-main-gradient: linear-gradient(135deg, #0A0D12 0%, #0E1117 100%);
            --bg-card: rgba(22, 27, 34, 0.92);
            --bg-card-hover: rgba(30, 36, 46, 0.95);
            --bg-soft: #1F2633;
            --border: rgba(46, 52, 64, 0.6);
            --border-light: rgba(26, 195, 174, 0.3);

            /* Text */
            --text-main: #F0F4F8;
            --text-secondary: #B0BEC5;
            --text-muted: #78909C;

            /* Effects */
            --glass: blur(20px) saturate(180%);
            --glass-dark: blur(24px) saturate(200%);
            --teal-glow: 0 0 25px rgba(26, 195, 174, 0.6);
            --purple-glow: 0 0 25px rgba(109, 82, 159, 0.5);
            --double-glow: 0 0 20px rgba(26, 195, 174, 0.5), 
                         0 0 40px rgba(109, 82, 159, 0.3);
        }

        /* ================= SMOOTH SCROLL ================= */
        html {
            scroll-behavior: smooth;
        }

        /* ================= ADVANCED ANIMATIONS ================= */
        @keyframes float {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(1deg); }
        }

        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes neuralPulse {
            0% { opacity: 0.4; transform: scale(0.95); }
            50% { opacity: 1; transform: scale(1.05); }
            100% { opacity: 0.4; transform: scale(0.95); }
        }

        @keyframes slideFade {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes typingDots {
            0%, 20% { opacity: 0.2; transform: translateY(0); }
            50% { opacity: 1; transform: translateY(-5px); }
            100% { opacity: 0.2; transform: translateY(0); }
        }

        @keyframes pulseBorder {
            0% { border-color: var(--border); }
            50% { border-color: var(--tanit-teal); }
            100% { border-color: var(--border); }
        }

        /* ================= GLOBAL ENHANCEMENTS ================= */
        * {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        body {
            background: var(--bg-main-gradient) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: var(--text-main) !important;
            line-height: 1.6 !important;
        }

        /* ================= ENHANCED HEADER ================= */
        .main-header {
            font-size: 3.5rem;
            font-weight: 900;
            background: var(--gradient-neuro);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.5rem;
            animation: float 6s ease-in-out infinite, 
                     gradientShift 8s ease infinite;
            background-size: 200% 200%;
            position: relative;
        }

        .main-header::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 200px;
            height: 4px;
            background: var(--gradient-medical);
            border-radius: 2px;
            animation: shimmer 3s infinite linear;
        }

        .tagline {
            text-align: center;
            color: var(--text-secondary);
            font-size: 1.1rem;
            letter-spacing: 0.5px;
            margin-bottom: 2.5rem;
            opacity: 0.9;
        }

        /* ================= NEURAL NETWORK BACKGROUND EFFECT ================= */
        .neural-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
            opacity: 0.05;
        }


        /* ================= ENHANCED CHAT CONTAINER ================= */
        .chat-container {
            position: relative;
            padding: 20px;
            border-radius: 24px;
            background: var(--bg-card);
            backdrop-filter: var(--glass);
            -webkit-backdrop-filter: var(--glass);
            border: 1px solid var(--border);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        }

        /* ================= CLEAN MESSAGE STYLING ================= */
        /* Chat message container */
        .chat-message {
            display: block;
            margin: 16px 0;
            padding: 16px;
            border-radius: 12px;
            transition: all 0.2s ease;
            width: 100%;
            box-sizing: border-box;
        }

        /* User message styling */
        .user-message {
            margin-left: auto;
            background: rgba(33, 150, 243, 0.15);
            border: 1px solid rgba(33, 150, 243, 0.3);
            border-radius: 12px 12px 4px 12px;
            max-width: 80%;
            text-align: left;
        }

        /* Bot message styling */
        .bot-message {
            margin-right: auto;
            background: rgba(26, 195, 174, 0.15);
            border: 1px solid rgba(26, 195, 174, 0.3);
            border-radius: 12px 12px 12px 4px;
            max-width: 80%;
            text-align: left;
        }

        /* Message text styling - SIMPLIFIED APPROACH */
        .message-content {
            font-size: 16px;
            line-height: 1.6;
            color: var(--text-main);
            margin: 0;
            padding: 0;
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* Reset all elements within message content to behave as normal flow */
        .message-content * {
            max-width: 100% !important;
            float: none !important;
            display: block !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            clear: both !important;
        }

        /* Specific element styling */
        .message-content p {
            margin: 0 0 12px 0;
            padding: 0;
            text-align: left;
        }

        .message-content p:last-child {
            margin-bottom: 0;
        }

        .message-content h1,
        .message-content h2,
        .message-content h3,
        .message-content h4 {
            margin: 16px 0 8px 0;
            padding: 0;
            color: var(--tanit-teal);
            font-weight: 600;
        }

        .message-content ul,
        .message-content ol {
            margin: 8px 0 8px 20px;
            padding: 0;
        }

        .message-content li {
            margin: 6px 0;
            padding: 0;
        }

        .message-content code {
            display: inline;
            background: rgba(0, 0, 0, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 14px;
        }

        .message-content pre {
            display: block;
            background: rgba(0, 0, 0, 0.2);
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 12px 0;
        }

        .message-content table {
            display: block;
            width: 100%;
            overflow-x: auto;
            margin: 12px 0;
            border-collapse: collapse;
        }

        .message-content td,
        .message-content th {
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 8px 12px;
            text-align: left;
        }

        /* ================= TYPING INDICATOR ================= */
        .typing-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 20px;
            background: var(--bg-card);
            border-radius: 18px;
            border: 1px solid var(--border-light);
            animation: pulseBorder 2s infinite;
            margin: 16px 0;
        }

        .typing-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--tanit-teal);
            animation: typingDots 1.4s infinite;
        }

        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        /* ================= ENHANCED INPUT ================= */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: rgba(15, 22, 36, 0.8) !important;
            color: var(--text-main) !important;
            border-radius: 16px !important;
            border: 2px solid var(--border) !important;
            padding: 1rem 1.2rem !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            backdrop-filter: var(--glass);
            -webkit-backdrop-filter: var(--glass);
            line-height: 1.5 !important;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: var(--tanit-teal) !important;
            box-shadow: var(--teal-glow) !important;
            transform: translateY(-2px);
            background: rgba(15, 22, 36, 0.95) !important;
        }

        /* ================= PREMIUM BUTTON ================= */
        .stButton > button {
            background: var(--gradient-medical) !important;
            color: white !important;
            border-radius: 16px !important;
            font-weight: 700 !important;
            padding: 0.85rem 2.2rem !important;
            border: none !important;
            font-size: 1rem !important;
            letter-spacing: 0.5px !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: 0 12px 30px rgba(26, 195, 174, 0.3) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            background-size: 200% 200% !important;
            animation: gradientShift 4s ease infinite !important;
            line-height: 1.5 !important;
        }

        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, 
                transparent, 
                rgba(255, 255, 255, 0.2), 
                transparent);
            transition: left 0.7s ease;
        }

        .stButton > button:hover {
            transform: translateY(-3px) scale(1.05) !important;
            box-shadow: 0 20px 40px rgba(26, 195, 174, 0.4) !important;
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        .stButton > button:active {
            transform: translateY(-1px) scale(0.98) !important;
        }

        /* ================= ENHANCED SIDEBAR ================= */
        section[data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, #0A0D12 0%, #0E1117 100%) !important;
            border-right: 1px solid var(--border) !important;
        }

        .sidebar-card {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 6px solid var(--tanit-teal);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
            backdrop-filter: var(--glass);
            -webkit-backdrop-filter: var(--glass);
            line-height: 1.6;
        }

        .sidebar-card:hover {
            transform: translateX(5px);
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.5);
            border-left-color: var(--tanit-purple);
        }

        /* ================= SIDEBAR STATUS INDICATORS ================= */
        .system-status-container {
            background: rgba(22, 27, 34, 0.7);
            border-radius: 18px;
            padding: 1.2rem;
            margin-bottom: 1.2rem;
            border: 1px solid var(--border);
            backdrop-filter: var(--glass);
            -webkit-backdrop-filter: var(--glass);
        }

        .status-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid var(--border);
        }

        .status-title {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            position: relative;
        }

        .status-icon::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        /* ALL STATUS COLORS USING TANIT TEAL VARIATIONS */
        .status-online {
            background: var(--tanit-teal-lighter);
            color: var(--tanit-teal-light);
            border: 1px solid rgba(26, 195, 174, 0.4);
        }

        .status-online .status-icon::before {
            background: var(--tanit-teal);
            box-shadow: 0 0 10px var(--tanit-teal);
            animation: statusPulse 2s infinite;
        }

        .status-offline {
            background: rgba(26, 195, 174, 0.1);
            color: rgba(106, 232, 216, 0.7);
            border: 1px solid rgba(26, 195, 174, 0.2);
        }

        .status-offline .status-icon::before {
            background: rgba(26, 195, 174, 0.5);
        }

        .status-warning {
            background: rgba(26, 195, 174, 0.15);
            color: var(--tanit-teal-light);
            border: 1px solid rgba(26, 195, 174, 0.3);
            animation: tealPulse 3s infinite;
        }

        .status-warning .status-icon::before {
            background: var(--tanit-teal);
            animation: neuralPulse 1.5s infinite;
        }

        .status-connecting {
            background: rgba(26, 195, 174, 0.15);
            color: var(--tanit-teal-light);
            border: 1px solid rgba(26, 195, 174, 0.3);
            animation: tealPulse 2s infinite;
        }

        .status-connecting .status-icon::before {
            background: var(--tanit-teal);
            animation: typingDots 1.4s infinite;
        }

        .status-error {
            background: rgba(26, 195, 174, 0.1);
            color: rgba(106, 232, 216, 0.6);
            border: 1px solid rgba(26, 195, 174, 0.2);
        }

        .status-error .status-icon::before {
            background: rgba(26, 195, 174, 0.4);
        }

        .status-details {
            margin-top: 0.8rem;
            padding-top: 0.8rem;
            border-top: 1px solid var(--border-light);
        }

        .status-detail-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .status-detail-label {
            color: var(--text-secondary);
        }

        .status-detail-value {
            color: var(--text-main);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }

        /* LLM-specific status - Using Tanit Teal */
        .llm-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.7rem;
            border-radius: 12px;
            background: rgba(26, 195, 174, 0.15);
            color: var(--tanit-teal-light);
            border: 1px solid rgba(26, 195, 174, 0.3);
            font-size: 0.8rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .llm-status-badge:hover {
            background: rgba(26, 195, 174, 0.25);
            transform: translateY(-1px);
        }

        .llm-status-badge::before {
            content: '';
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--tanit-teal);
            display: inline-block;
            animation: tealGlow 2s infinite;
        }

        .status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }

        .status-label {
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        /* Enhanced status value styling */
        .status-value-teal {
            color: var(--tanit-teal-light);
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            background: rgba(26, 195, 174, 0.1);
            border-radius: 6px;
            border: 1px solid rgba(26, 195, 174, 0.2);
        }

        /* Performance indicators */
        .performance-indicator {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }

        .performance-excellent {
            background: rgba(26, 195, 174, 0.2);
            color: var(--tanit-teal-light);
            border: 1px solid rgba(26, 195, 174, 0.3);
        }

        .performance-good {
            background: rgba(26, 195, 174, 0.15);
            color: rgba(106, 232, 216, 0.9);
            border: 1px solid rgba(26, 195, 174, 0.25);
        }

        .performance-fair {
            background: rgba(26, 195, 174, 0.1);
            color: rgba(106, 232, 216, 0.7);
            border: 1px solid rgba(26, 195, 174, 0.2);
        }

        /* ================= PROGRESS INDICATORS ================= */
        .stProgress > div > div > div {
            background: var(--gradient-medical) !important;
            background-size: 200% 100% !important;
            animation: gradientShift 2s ease infinite !important;
        }

        /* ================= TOOLTIPS ================= */
        [data-tooltip] {
            position: relative;
            cursor: help;
        }

        [data-tooltip]:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 8px 12px;
            background: var(--bg-card);
            color: var(--text-main);
            border-radius: 8px;
            font-size: 0.9rem;
            white-space: nowrap;
            z-index: 1000;
            border: 1px solid var(--border);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            animation: slideFade 0.3s ease;
            line-height: 1.4;
        }

        /* ================= SCROLLBAR STYLING ================= */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-soft);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--gradient-medical);
            border-radius: 10px;
            border: 2px solid var(--bg-soft);
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--tanit-teal-light);
        }

        /* ================= LOADING STATES ================= */
        .loading-pulse {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--tanit-teal);
            animation: neuralPulse 1.5s ease-in-out infinite;
            margin: 0 4px;
        }

        /* ================= BADGES & TAGS ================= */
        .medical-badge {
            display: inline-block;
            padding: 4px 12px;
            background: rgba(26, 195, 174, 0.15);
            color: var(--tanit-teal-light);
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(26, 195, 174, 0.3);
            margin: 2px;
            animation: pulseBorder 3s infinite;
            line-height: 1.4;
        }

        /* ================= STREAMLIT SPECIFIC FIXES ================= */
        .stApp {
            background: var(--bg-main-gradient) !important;
        }
        
        /* Ensure text colors are consistent */
        p, h1, h2, h3, h4, h5, h6, span, div, label {
            color: var(--text-main) !important;
            line-height: 1.6 !important;
        }
        
        /* Fix for Streamlit containers */
        [data-testid="stHorizontalBlock"] {
            align-items: flex-start !important;
        }
        
        /* Ensure markdown content displays properly */
        .stMarkdown {
            width: 100%;
        }
        
        /* Fix for chat message containers */
        [data-testid="chatMessage"] {
            width: 100%;
            max-width: 100%;
        }

        /* ================= RESPONSIVE ================= */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2.5rem;
            }
            
            .chat-message {
                padding: 12px;
                margin: 12px 0;
            }
            
            .user-message,
            .bot-message {
                max-width: 90%;
            }
            
            .message-content {
                font-size: 15px;
                line-height: 1.5;
            }
            
            .typing-indicator {
                padding: 10px 16px;
            }
            
            .system-status-container {
                padding: 1rem;
            }
            
            .status-indicator {
                font-size: 0.8rem;
                padding: 0.3rem 0.6rem;
            }
        }

        /* ================= STATUS COLOR FIXES ================= */
        /* Override Streamlit's default status colors */
        .stSuccess, .stAlert, .stWarning, .stError {
            background-color: transparent !important;
            border: none !important;
        }
        
        .stSuccess > div, .stAlert > div, .stWarning > div, .stError > div {
            background-color: transparent !important;
        }
                    


                    

        </style>
                    
                    
        """, unsafe_allow_html=True)

    @staticmethod
    def typing_animation():
        """Returns a modern typing indicator"""
        return """
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span style="margin-left: 8px; color: var(--text-secondary); font-size: 0.9rem;">
                AI is thinking...
            </span>
        </div>
        """

    @staticmethod
    def badge(text, color="teal"):
        """Create a medical-themed badge"""
        colors = {
            "teal": "var(--tanit-teal)",
            "purple": "var(--tanit-purple)",
            "warning": "var(--warning)",
            "success": "var(--success)"
        }
        return f'<span class="medical-badge" style="border-color: {colors.get(color, colors["teal"])}">{text}</span>'

    @staticmethod
    def message(text, is_user=False):
        """Format chat messages with proper styling"""
        message_class = "user-message" if is_user else "bot-message"
        return f"""
        <div class="chat-message {message_class}">
            <div class="message-content">
                {text}
            </div>
        </div>
        """