from langchain_community.chat_message_histories import ChatMessageHistory

class SessionStore:

    def __init__(self):
        self.sessions = {}

    def get_session(self, session_id):

        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": ChatMessageHistory(),
                "summary": ""
            }

        return self.sessions[session_id]


session_store = SessionStore()