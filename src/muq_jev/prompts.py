TYPE_LABELS = {'genre': 'GENRE', 'mood': 'MOOD', 'instrument': 'INSTRUMENT', 'solo_instrument': 'SOLO INSTRUMENT', 'vocal_gender': 'VOCAL GENDER', 'voice_presence': 'VOICE PRESENCE', 'top50': 'TOP50 TAG'}
QUESTIONS = {'genre': 'Which genre best describes this music?', 'mood': 'Which mood or theme best describes this music?', 'instrument': 'Which instruments are present in this audio?', 'solo_instrument': 'Which instrument is the solo instrument in this clip?', 'vocal_gender': 'What is the vocal gender in this audio?', 'voice_presence': 'Is there a singing voice in this audio?', 'top50': 'Which of these tags describe this music?'}

def readable(candidate: str) -> str:
    text = str(candidate)
    for prefix in ('genre---', 'mood/theme---', 'instrument---'):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.replace('---', ' / ').replace('_', ' ').strip()
    return text or str(candidate)

def build_prompt(task: str, candidate: str) -> str:
    return f"{TYPE_LABELS[task]} | {QUESTIONS[task]} | Candidate: {readable(candidate)}"
