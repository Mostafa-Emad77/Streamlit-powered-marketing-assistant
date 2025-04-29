import streamlit as st # for UI
from firecrawl import FirecrawlApp,ScrapeOptions # for web crawling
from langchain.memory import ConversationBufferMemory # for memory management
from huggingface_hub import InferenceClient # for image generation
import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
FIRECRAWL_KEY = os.getenv("FIRECRAWL_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
TARGET_URL = "https://34ml.com/"
MAX_PAGES = 50
SITE_CONTENT_FILE = "site_content.txt"
ANALYSIS_FILE = "34ml_analysis.md"

def save_markdown_analysis(content):
    """Save the analyzed content as a Markdown file"""
    try:
        with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error saving markdown analysis: {str(e)}")
        return False

def load_markdown_analysis():
    """Load the analyzed content from the Markdown file"""
    try:
        if os.path.exists(ANALYSIS_FILE):
            with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
                return f.read()
        return None
    except Exception as e:
        print(f"Error loading markdown analysis: {str(e)}")
        return None

# --- Content Templates ---
PLATFORM_TEMPLATES = {
    "linkedin": {
        "system_prompt": "You are a professional LinkedIn content creator. Create ONE optimized post that combines company achievements, client success stories, and industry insights. Focus on business value and engaging storytelling. Use appropriate hashtags.",
        "max_length": 3000
    },
    "facebook": {
        "system_prompt": "You are a conversational Facebook content creator. Create ONE engaging post that combines client appreciation, success stories, and company culture. Be warm and professional. Include relevant emojis and hashtags.",
        "max_length": 63206
    },
    "instagram": {
        "system_prompt": "You are an Instagram content creator. Create ONE impactful post that showcases company achievements, client success, and team culture. Be visual and engaging. Use emojis and hashtags effectively.",
        "max_length": 2200
    },
    "blog": {
        "system_prompt": "You are a professional blog writer. Create ONE comprehensive article that combines industry insights, success stories, and actionable advice. Include clear headings and examples.",
        "max_length": 10000
    },
    "email": {
        "system_prompt": "You are an email marketing specialist. Create ONE compelling email that combines company news, client success stories, and valuable insights. Focus on driving action while maintaining professionalism.",
        "max_length": 5000
    }
}

# --- Content History ---
CONTENT_HISTORY_FILE = "content_history.json"

def load_content_history():
    print("Debug: Attempting to load content history")
    try:
        if os.path.exists(CONTENT_HISTORY_FILE):
            with open(CONTENT_HISTORY_FILE, 'r') as f:
                history = json.load(f)
                print(f"Debug: Successfully loaded history with {len(history)} entries")
                return history
        else:
            print("Debug: No history file found, creating new one")
            with open(CONTENT_HISTORY_FILE, 'w') as f:
                json.dump([], f)
            return []
    except Exception as e:
        print(f"Debug: Error loading content history - {str(e)}")
        return []

def save_content_history(platform, content):
    print(f"Debug: Attempting to save content for {platform}")
    try:
        history = load_content_history()
        print(f"Debug: Loaded existing history with {len(history)} entries")
        entry = {
            "platform": platform,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        history.append(entry)
        print("Debug: Added new entry to history")
        with open(CONTENT_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        print("Debug: Successfully saved updated history")
        return True
    except Exception as e:
        print(f"Debug: Error saving content history - {str(e)}")
        return False

# --- Firecrawl Setup ---
@st.cache_resource(show_spinner="Crawling the website ...") 
def crawl_site():
    app = FirecrawlApp(api_key=FIRECRAWL_KEY)
    crawl_result = app.crawl_url(
        url=TARGET_URL,
        limit=MAX_PAGES,
        scrape_options=ScrapeOptions(formats=['markdown', 'html']),
    )
    pages = getattr(crawl_result, "data", [])
    all_content = []
    for page in pages:
        content = getattr(page, "markdown", "") if hasattr(page, "markdown") else page.get("markdown", "")
        if content and content.strip():
            all_content.append(content)
    full_content = "\n\n".join(all_content)
    return full_content

def analyze_site_content(content):
    """
    Use Gemini to analyze and structure the site content into detailed Markdown format
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        
        # Enhanced prompt requesting detailed Markdown format
        analysis_prompt = """Analyze the following website content and create a comprehensive structured Markdown document.
        Extract and organize all available information into these categories:

        # Company Overview
        ## Core Business
        - Main focus areas
        - Key expertise
        - Industry position

        ## Mission & Vision
        - Mission statement
        - Long-term vision
        - Company goals

        ## Values & Philosophy
        - Core values
        - Company culture
        - Business approach

        # Services & Solutions
        ## Main Service Categories
        - List and describe each service category
        - Key features and benefits
        - Target markets

        ## Technical Capabilities
        - Development expertise
        - Technologies used
        - Methodologies

        ## Product Portfolio
        - Notable products
        - Case studies
        - Success metrics

        # Customer Base & Market Presence
        ## Industries Served
        - Primary industries
        - Sector expertise
        - Market segments

        ## Client Types
        - Business sizes
        - Geographic regions
        - Client profiles

        ## Notable Clients & Projects
        - Major clients
        - Significant projects
        - Achievement highlights

        # Success Stories & Testimonials
        ## Case Studies
        - Detailed project examples
        - Problem-solution scenarios
        - Results and impacts

        ## Client Testimonials
        - Client feedback
        - Success metrics
        - Client satisfaction highlights

        ## Key Achievements
        - Industry recognition
        - Awards
        - Notable milestones

        # Content & Knowledge Sharing
        ## Blog Articles
        - Main topics
        - Key insights
        - Recent publications

        ## Technical Resources
        - Guides
        - Tutorials
        - Best practices

        ## Industry Insights
        - Market analysis
        - Trend discussions
        - Expert perspectives

        # Brand Identity
        ## Voice & Tone
        - Communication style
        - Brand personality
        - Key messaging themes

        ## Visual Identity
        - Design principles
        - Brand elements
        - Style guidelines

        ## Market Positioning
        - Unique value proposition
        - Competitive advantages
        - Market differentiators

        # Technical Expertise
        ## Technology Stack
        - Programming languages
        - Frameworks
        - Tools & platforms

        ## Development Practices
        - Methodologies
        - Best practices
        - Quality standards

        ## Innovation Focus
        - R&D areas
        - Emerging technologies
        - Innovation initiatives

        # Team & Culture
        ## Company Values
        - Core principles
        - Team ethics
        - Cultural pillars

        ## Work Environment
        - Office culture
        - Work approach
        - Team dynamics

        ## Growth & Development
        - Learning opportunities
        - Career growth
        - Professional development

        Content to analyze: {content}

        Important instructions:
        1. Extract specific, factual information from the content
        2. Use bullet points for clarity
        3. Include direct quotes where relevant
        4. Maintain proper Markdown hierarchy
        5. Be comprehensive but concise
        6. Focus on concrete details rather than general statements
        7. Include metrics and numbers where available
        8. Highlight unique aspects and differentiators
        """
        
        # Trim content to fit within limits but process in chunks if needed
        content_chunks = [content[i:i+50000] for i in range(0, len(content), 50000)]
        all_analyses = []
        
        for chunk in content_chunks:
            data = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": analysis_prompt.format(content=chunk)}]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.8,
                    "topK": 40,
                    "maxOutputTokens": 2048
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            response_json = response.json()
            if 'candidates' in response_json:
                all_analyses.append(response_json['candidates'][0]['content']['parts'][0]['text'])
        
        # Combine and deduplicate the analyses
        combined_analysis = "\n\n".join(all_analyses)
        
        # Save the markdown content to a file
        save_markdown_analysis(combined_analysis)
        
        return {"markdown_content": combined_analysis}
            
    except Exception as e:
        print(f"Error analyzing site content: {str(e)}")
        return None

def load_or_scrape_site():
    """Load or scrape site content and analyze it"""
    if os.path.exists(SITE_CONTENT_FILE):
        with open(SITE_CONTENT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = crawl_site()
        with open(SITE_CONTENT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
    
    # Try to load existing analysis first
    existing_analysis = load_markdown_analysis()
    if existing_analysis:
        return {
            "raw_content": content,
            "knowledge_base": {
                "markdown_content": existing_analysis
            }
        }
    
    # If no existing analysis, create new one
    knowledge_base = analyze_site_content(content)
    if knowledge_base and knowledge_base.get("markdown_content"):
        save_markdown_analysis(knowledge_base["markdown_content"])
    return {"raw_content": content, "knowledge_base": knowledge_base}

# --- Memory Management ---
def get_conversation_memory():
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="input",
            output_key="output"
        )
    return st.session_state.memory

def save_to_memory(input_text, output_text):
    memory = get_conversation_memory()
    memory.save_context({"input": input_text}, {"output": output_text})

def ask_gemini(messages, platform="general"):
    try:
        # Get the knowledge base and memory
        knowledge_base = st.session_state.get("knowledge_base")
        memory = get_conversation_memory()
        
        if not knowledge_base:
            return "Error: Knowledge base not initialized"

        # Use markdown content if available
        markdown_content = knowledge_base.get("markdown_content", "")
        
        # Include platform-specific instructions
        template = PLATFORM_TEMPLATES.get(platform, {
            "system_prompt": "You are a helpful assistant that always responds in English.",
            "max_length": 5000
        })
        
        # Create focused context using markdown content and memory
        context = f"""
        Use this analyzed information about 34ML to inform your responses:

        {markdown_content}

        Previous conversation context:
        {memory.buffer}

        Maintain the company's tone of voice and refer to relevant success stories and services when appropriate.
        """
        
        system_message = f"{template['system_prompt']}\n\n{context}"
        
        # Add content history context
        history = load_content_history()
        recent_content = [h for h in history if h["platform"] == platform][-5:] if history else []
        if recent_content:
            system_message += "\n\nRecent content history (avoid repetition):\n" + "\n".join(
                [f"- {h['content'][:100]}..." for h in recent_content]
            )

        # Format messages for Gemini API
        messages_content = []
        for message in messages:
            if message["role"] != "system":
                messages_content.append({
                    "role": "user" if message["role"] == "user" else "model",
                    "parts": [{"text": message["content"]}]
                })

        # Prepare the API request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": system_message}]
                },
                *messages_content
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 2048
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        response_json = response.json()
        if 'candidates' in response_json:
            generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
            # Save to conversation memory only
            if messages and messages[-1]["role"] == "user":
                save_to_memory(messages[-1]["content"], generated_text)
            return generated_text
        else:
            print(f"Unexpected response format: {response_json}")
            return "Sorry, I couldn't generate a response due to an unexpected API response format."
            
    except requests.exceptions.RequestException as e:
        print(f"API request error: {str(e)}")
        if hasattr(e.response, 'json'):
            print(f"Error details: {e.response.json()}")
        return f"Sorry, I couldn't generate a response. API error: {str(e)}"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return "Sorry, I couldn't generate a response due to an unexpected error."

# Add this new function after the ask_gemini function
def generate_topic_suggestions(platform, past_topics):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        
        prompt = f"""As a content strategist for {platform}, suggest 5 unique content ideas that haven't been covered in these past topics:

        Past topics:
        {past_topics}

        Generate fresh, innovative ideas that align with {platform}'s content style while avoiding any topics similar to the past ones.
        Return only the numbered list of suggestions, one per line.
        """
        
        data = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.9,
                "topP": 0.8,
                "maxOutputTokens": 1024
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        if 'candidates' in response.json():
            suggestions = response.json()['candidates'][0]['content']['parts'][0]['text'].strip().split('\n')
            return [s.strip('123456789. ') for s in suggestions if s.strip()]
        return []
    except Exception as e:
        print(f"Error generating suggestions: {str(e)}")
        return []

# --- Hugging Face Setup ---
hf_client = InferenceClient(
    provider="fal-ai",
    api_key=HF_API_KEY,
)

def generate_image(prompt):
    image = hf_client.text_to_image(
        prompt,
        model="HiDream-ai/HiDream-I1-Full",
    )
    return image

# --- Streamlit UI ---
st.set_page_config(page_title="34ML AI Assistant", page_icon="🤖")
st.title("🤖 34ML Social Media Assistant")
st.info("Generate content for various platforms. Try: 'Write a LinkedIn post about our services' or 'Create a blog post about AI integration'")

# Initialize session state
if "site_content" not in st.session_state:
    with st.spinner("Analyzing website content..."):
        content_data = load_or_scrape_site()
        st.session_state["site_content"] = content_data["raw_content"]
        st.session_state["knowledge_base"] = content_data["knowledge_base"]

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = None
if "current_response" not in st.session_state:
    st.session_state.current_response = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Platform selection
platform = st.selectbox("Select Platform", ["linkedin", "facebook", "instagram", "blog", "email"])

# Content type suggestions based on platform
content_suggestions = {
    "linkedin": ["Comprehensive Business Update"],
    "facebook": ["Community & Success Spotlight"],
    "instagram": ["Visual Success Story"],
    "blog": ["In-depth Industry Analysis"],
    "email": ["Client Success Newsletter"]
}

content_type = st.selectbox(f"Content Type for {platform.title()}", content_suggestions[platform])

# Load history for content suggestions
history = load_content_history()
print("Debug: Loaded content history length:", len(history))
platform_history = [h for h in history if h["platform"] == platform][-10:] if history else []

# Show content suggestions based on history
if platform_history:
    st.sidebar.markdown("### Content Ideas Based on History")
    topics_used = [entry["content"][:100] for entry in platform_history]
    
    # Get dynamic suggestions using Gemini
    suggested_topics = generate_topic_suggestions(platform, "\n".join(topics_used))
    
    st.sidebar.markdown("#### AI-Generated Topic Suggestions:")
    for topic in suggested_topics:
        st.sidebar.markdown(f"- {topic}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Image Generation Section
st.markdown("---")
st.subheader("🎨 Image Generation")
image_prompt = st.text_input("Enter a prompt to generate an image:", placeholder="E.g., A professional office workspace with modern tech devices")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        try:
            generated_image = generate_image(image_prompt)
            st.image(generated_image, caption=f"Generated image for: {image_prompt}")
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")

# Chat input
if prompt := st.chat_input(f"What would you like to create for {platform}?"):
    # Add platform context to the prompt
    enhanced_prompt = f"Create {content_type} content for {platform}: {prompt}"
    
    st.session_state.messages.append({"role": "user", "content": enhanced_prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate content
    with st.chat_message("assistant"):
        response = ask_gemini(st.session_state.messages, platform)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Auto-save the response immediately
        save_content_history(platform, response)
        print("Debug: Auto-saved response")

        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Approve", key=f"approve_{len(st.session_state.messages)}"):
                st.success("Content approved!")
        
        with col2:
            if st.button("❌ Decline", key=f"decline_{len(st.session_state.messages)}"):
                st.warning("Content declined.")

# Show approved content history
if platform_history:
    st.sidebar.markdown("### Content History")
    for entry in reversed(platform_history[-5:]):
        with st.sidebar.expander(f"Post from {entry['timestamp'][:10]}"):
            st.markdown(entry['content'])
