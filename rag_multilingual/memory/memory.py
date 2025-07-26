from config.settings import SHORT_TERM_MEMORY_SIZE


class Memory:
    def __init__(self, max_history=SHORT_TERM_MEMORY_SIZE):
        self.history = []
        self.max_history = max_history

    def add(self, user, query, answer):
        self.history.append({'user': user, 'query': query, 'answer': answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):
        return "\n".join([
            f"User: {h['query']}\nBot: {h['answer']}" for h in self.history
        ])


short_term_memory = Memory()
# -*- coding: utf-8 -*-

