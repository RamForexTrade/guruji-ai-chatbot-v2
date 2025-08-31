import re
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Teaching:
    """Structure to hold teaching data"""
    number: str
    title: str
    date: str
    location: str
    topics: List[str]
    keywords: List[str]
    problem_categories: List[str]
    emotional_states: List[str]
    life_situations: List[str]
    content: str
    
    def to_dict(self) -> Dict:
        """Convert teaching to dictionary for vector storage"""
        return {
            'number': self.number,
            'title': self.title,
            'date': self.date,
            'location': self.location,
            'topics': self.topics,
            'keywords': self.keywords,
            'problem_categories': self.problem_categories,
            'emotional_states': self.emotional_states,
            'life_situations': self.life_situations,
            'content': self.content,
            'full_text': self.get_full_text()
        }
    
    def get_full_text(self) -> str:
        """Get full searchable text including metadata"""
        metadata_text = f"""
        Teaching #{self.number}: {self.title}
        Date: {self.date}
        Location: {self.location}
        Topics: {', '.join(self.topics)}
        Keywords: {', '.join(self.keywords)}
        Problem Categories: {', '.join(self.problem_categories)}
        Emotional States: {', '.join(self.emotional_states)}
        Life Situations: {', '.join(self.life_situations)}
        """
        return f"{metadata_text}\n\nContent:\n{self.content}"

class DocumentProcessor:
    """Process markdown files containing Sri Sri Ravi Shankar's teachings"""
    
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        
    def load_all_teachings(self) -> List[Teaching]:
        """Load all teachings from markdown files"""
        all_teachings = []
        
        for md_file in self.knowledge_base_path.glob("*.md"):
            teachings = self.parse_markdown_file(md_file)
            all_teachings.extend(teachings)
            
        print(f"Loaded {len(all_teachings)} teachings from {len(list(self.knowledge_base_path.glob('*.md')))} files")
        return all_teachings
    
    def parse_markdown_file(self, file_path: Path) -> List[Teaching]:
        """Parse a single markdown file and extract teachings"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by teaching sections
        teaching_sections = re.split(r'\n## Teaching #', content)
        teachings = []
        
        for i, section in enumerate(teaching_sections):
            if i == 0:  # Skip the header section
                continue
                
            # Add back the "## Teaching #" prefix
            section = "## Teaching #" + section
            teaching = self.parse_teaching_section(section)
            if teaching:
                teachings.append(teaching)
        
        return teachings
    
    def parse_teaching_section(self, section: str) -> Teaching:
        """Parse a single teaching section"""
        try:
            lines = section.strip().split('\n')
            
            # Extract teaching number and title
            title_line = lines[0].replace('## Teaching #', '').strip()
            if ':' in title_line:
                number, title = title_line.split(':', 1)
                number = number.strip()
                title = title.strip()
            else:
                number = title_line.strip()
                title = "Untitled"
            
            # Initialize variables
            date = "Not specified"
            location = "Not specified"
            topics = []
            keywords = []
            problem_categories = []
            emotional_states = []
            life_situations = []
            content = ""
            
            content_started = False
            
            for line in lines[1:]:
                line = line.strip()
                
                if line.startswith('**Date:**'):
                    date = line.replace('**Date:**', '').strip()
                elif line.startswith('**Location:**'):
                    location = line.replace('**Location:**', '').strip()
                elif line.startswith('**Topics:**'):
                    topics_str = line.replace('**Topics:**', '').strip()
                    topics = [t.strip() for t in topics_str.split(',') if t.strip()]
                elif line.startswith('**Keywords:**'):
                    keywords_str = line.replace('**Keywords:**', '').strip()
                    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                elif line.startswith('**Problem Categories:**'):
                    prob_str = line.replace('**Problem Categories:**', '').strip()
                    problem_categories = [p.strip() for p in prob_str.split(',') if p.strip()]
                elif line.startswith('**Emotional States:**'):
                    emot_str = line.replace('**Emotional States:**', '').strip()
                    emotional_states = [e.strip() for e in emot_str.split(',') if e.strip()]
                elif line.startswith('**Life Situations:**'):
                    life_str = line.replace('**Life Situations:**', '').strip()
                    life_situations = [l.strip() for l in life_str.split(',') if l.strip()]
                elif line.startswith('### Content:'):
                    content_started = True
                elif content_started and line and line != '---':
                    if content:
                        content += "\n" + line
                    else:
                        content = line
            
            return Teaching(
                number=number,
                title=title,
                date=date,
                location=location,
                topics=topics,
                keywords=keywords,
                problem_categories=problem_categories,
                emotional_states=emotional_states,
                life_situations=life_situations,
                content=content.strip()
            )
            
        except Exception as e:
            print(f"Error parsing teaching section: {e}")
            return None
    
    def search_teachings_by_metadata(self, teachings: List[Teaching], 
                                   query_topics: List[str] = None,
                                   query_emotions: List[str] = None,
                                   query_problems: List[str] = None,
                                   query_situations: List[str] = None) -> List[Teaching]:
        """Search teachings based on metadata filters"""
        filtered_teachings = []
        
        for teaching in teachings:
            score = 0
            
            # Check topics
            if query_topics:
                topic_matches = sum(1 for topic in query_topics 
                                  if any(topic.lower() in t.lower() for t in teaching.topics))
                score += topic_matches * 3
            
            # Check emotional states
            if query_emotions:
                emotion_matches = sum(1 for emotion in query_emotions 
                                    if any(emotion.lower() in e.lower() for e in teaching.emotional_states))
                score += emotion_matches * 2
            
            # Check problem categories
            if query_problems:
                problem_matches = sum(1 for problem in query_problems 
                                    if any(problem.lower() in p.lower() for p in teaching.problem_categories))
                score += problem_matches * 2
            
            # Check life situations
            if query_situations:
                situation_matches = sum(1 for situation in query_situations 
                                      if any(situation.lower() in s.lower() for s in teaching.life_situations))
                score += situation_matches * 2
            
            if score > 0:
                filtered_teachings.append((teaching, score))
        
        # Sort by score and return teachings
        filtered_teachings.sort(key=lambda x: x[1], reverse=True)
        return [teaching for teaching, score in filtered_teachings]