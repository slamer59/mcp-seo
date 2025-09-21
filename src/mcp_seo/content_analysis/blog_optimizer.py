"""
Blog content optimization utilities for SEO analysis.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class ContentQuality(Enum):
    """Content quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"


@dataclass
class ReadabilityScore:
    """Readability analysis results."""
    flesch_reading_ease: float
    flesch_grade_level: float
    gunning_fog: float
    smog_index: float
    automated_readability_index: float
    overall_grade: str
    recommendations: List[str]


@dataclass
class KeywordDensity:
    """Keyword density analysis."""
    keyword: str
    count: int
    density: float
    target_density: float
    status: str  # 'optimal', 'low', 'high'


@dataclass
class ContentStructure:
    """Content structure analysis."""
    word_count: int
    paragraph_count: int
    sentence_count: int
    avg_words_per_sentence: float
    avg_sentences_per_paragraph: float
    heading_structure: Dict[str, int]
    list_count: int
    image_count: int
    link_count: int


@dataclass
class BlogOptimizationResult:
    """Complete blog optimization analysis."""
    content_quality: ContentQuality
    readability: ReadabilityScore
    keyword_analysis: List[KeywordDensity]
    content_structure: ContentStructure
    seo_score: int
    recommendations: List[str]


class BlogContentOptimizer:
    """Advanced blog content optimizer for SEO."""

    def __init__(self):
        self.target_word_count_min = 300
        self.target_word_count_max = 2500
        self.optimal_keyword_density = (1.0, 3.0)  # 1-3%
        self.optimal_readability_score = (60, 80)  # Flesch Reading Ease

    def analyze_blog_content(
        self,
        content: str,
        target_keywords: List[str],
        title: Optional[str] = None,
        meta_description: Optional[str] = None
    ) -> BlogOptimizationResult:
        """Perform comprehensive blog content analysis."""

        # Basic content structure analysis
        structure = self._analyze_content_structure(content)

        # Readability analysis
        readability = self._analyze_readability(content)

        # Keyword analysis
        keyword_analysis = self._analyze_keywords(content, target_keywords)

        # Overall content quality assessment
        quality = self._assess_content_quality(structure, readability, keyword_analysis)

        # SEO score calculation
        seo_score = self._calculate_seo_score(structure, readability, keyword_analysis, title, meta_description)

        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            structure, readability, keyword_analysis, quality, title, meta_description
        )

        return BlogOptimizationResult(
            content_quality=quality,
            readability=readability,
            keyword_analysis=keyword_analysis,
            content_structure=structure,
            seo_score=seo_score,
            recommendations=recommendations
        )

    def _analyze_content_structure(self, content: str) -> ContentStructure:
        """Analyze the structural elements of content."""

        # Clean content for analysis
        clean_content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags

        # Word count
        words = re.findall(r'\b\w+\b', clean_content.lower())
        word_count = len(words)

        # Sentence count
        sentences = re.split(r'[.!?]+', clean_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        # Paragraph count (assuming double newlines or <p> tags)
        paragraphs = re.split(r'\n\s*\n|<p[^>]*>', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        paragraph_count = len(paragraphs)

        # Average calculations
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        avg_sentences_per_paragraph = sentence_count / paragraph_count if paragraph_count > 0 else 0

        # Heading structure
        heading_structure = {}
        for level in range(1, 7):
            h_pattern = f'<h{level}[^>]*>.*?</h{level}>'
            heading_structure[f'h{level}'] = len(re.findall(h_pattern, content, re.IGNORECASE | re.DOTALL))

        # Lists
        list_count = len(re.findall(r'<[uo]l[^>]*>', content, re.IGNORECASE))

        # Images
        image_count = len(re.findall(r'<img[^>]*>', content, re.IGNORECASE))

        # Links
        link_count = len(re.findall(r'<a[^>]*href', content, re.IGNORECASE))

        return ContentStructure(
            word_count=word_count,
            paragraph_count=paragraph_count,
            sentence_count=sentence_count,
            avg_words_per_sentence=avg_words_per_sentence,
            avg_sentences_per_paragraph=avg_sentences_per_paragraph,
            heading_structure=heading_structure,
            list_count=list_count,
            image_count=image_count,
            link_count=link_count
        )

    def _analyze_readability(self, content: str) -> ReadabilityScore:
        """Analyze content readability using multiple metrics."""

        # Clean content
        clean_content = re.sub(r'<[^>]+>', '', content)

        # Basic counts
        words = re.findall(r'\b\w+\b', clean_content.lower())
        sentences = re.split(r'[.!?]+', clean_content)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_count = len(words)
        sentence_count = len(sentences)

        # Syllable count (simplified)
        syllable_count = self._count_syllables(' '.join(words))

        # Average calculations
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        avg_syllables_per_word = syllable_count / word_count if word_count > 0 else 0

        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        flesch_reading_ease = max(0, min(100, flesch_reading_ease))

        # Flesch-Kincaid Grade Level
        flesch_grade_level = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
        flesch_grade_level = max(0, flesch_grade_level)

        # Gunning Fog Index (simplified)
        complex_words = [word for word in words if self._count_syllables(word) >= 3]
        complex_word_percentage = len(complex_words) / word_count * 100 if word_count > 0 else 0
        gunning_fog = 0.4 * (avg_words_per_sentence + complex_word_percentage)

        # SMOG Index (simplified)
        if sentence_count >= 30:
            complex_words_per_sentence = len(complex_words) / sentence_count
            smog_index = 1.043 * math.sqrt(complex_words_per_sentence * 30) + 3.1291
        else:
            smog_index = flesch_grade_level  # Fallback

        # Automated Readability Index
        characters = len(re.sub(r'\s', '', clean_content))
        ari = (4.71 * (characters / word_count)) + (0.5 * avg_words_per_sentence) - 21.43 if word_count > 0 else 0

        # Overall grade
        if flesch_reading_ease >= 90:
            overall_grade = "Very Easy"
        elif flesch_reading_ease >= 80:
            overall_grade = "Easy"
        elif flesch_reading_ease >= 70:
            overall_grade = "Fairly Easy"
        elif flesch_reading_ease >= 60:
            overall_grade = "Standard"
        elif flesch_reading_ease >= 50:
            overall_grade = "Fairly Difficult"
        elif flesch_reading_ease >= 30:
            overall_grade = "Difficult"
        else:
            overall_grade = "Very Difficult"

        # Recommendations
        recommendations = []
        if flesch_reading_ease < 60:
            recommendations.append("Use shorter sentences to improve readability")
            recommendations.append("Replace complex words with simpler alternatives")
        if avg_words_per_sentence > 20:
            recommendations.append("Break up long sentences into shorter ones")
        if complex_word_percentage > 15:
            recommendations.append("Reduce the use of complex, multi-syllable words")

        return ReadabilityScore(
            flesch_reading_ease=flesch_reading_ease,
            flesch_grade_level=flesch_grade_level,
            gunning_fog=gunning_fog,
            smog_index=smog_index,
            automated_readability_index=ari,
            overall_grade=overall_grade,
            recommendations=recommendations
        )

    def _analyze_keywords(self, content: str, target_keywords: List[str]) -> List[KeywordDensity]:
        """Analyze keyword density and optimization."""

        # Clean content
        clean_content = re.sub(r'<[^>]+>', '', content.lower())
        words = re.findall(r'\b\w+\b', clean_content)
        total_words = len(words)

        keyword_analysis = []

        for keyword in target_keywords:
            keyword_lower = keyword.lower()

            # Count exact phrase matches
            exact_matches = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', clean_content))

            # Count individual word matches (for multi-word keywords)
            keyword_words = keyword_lower.split()
            individual_matches = 0
            for kw in keyword_words:
                individual_matches += words.count(kw)

            # Use exact matches for calculation, fall back to individual if no exact matches
            count = exact_matches if exact_matches > 0 else individual_matches

            # Calculate density
            density = (count / total_words * 100) if total_words > 0 else 0

            # Determine status
            target_min, target_max = self.optimal_keyword_density
            if density < target_min:
                status = "low"
            elif density > target_max:
                status = "high"
            else:
                status = "optimal"

            keyword_analysis.append(KeywordDensity(
                keyword=keyword,
                count=count,
                density=density,
                target_density=(target_min + target_max) / 2,
                status=status
            ))

        return keyword_analysis

    def _assess_content_quality(
        self,
        structure: ContentStructure,
        readability: ReadabilityScore,
        keyword_analysis: List[KeywordDensity]
    ) -> ContentQuality:
        """Assess overall content quality."""

        score = 0
        max_score = 100

        # Word count assessment (25 points)
        if self.target_word_count_min <= structure.word_count <= self.target_word_count_max:
            score += 25
        elif structure.word_count >= self.target_word_count_min:
            score += 20
        elif structure.word_count >= 200:
            score += 15
        else:
            score += 5

        # Readability assessment (25 points)
        if self.optimal_readability_score[0] <= readability.flesch_reading_ease <= self.optimal_readability_score[1]:
            score += 25
        elif readability.flesch_reading_ease >= 50:
            score += 20
        elif readability.flesch_reading_ease >= 30:
            score += 15
        else:
            score += 10

        # Structure assessment (25 points)
        structure_score = 0
        if structure.heading_structure.get('h1', 0) >= 1:
            structure_score += 5
        if sum(structure.heading_structure.values()) >= 3:
            structure_score += 5
        if structure.paragraph_count >= 3:
            structure_score += 5
        if structure.image_count >= 1:
            structure_score += 5
        if structure.link_count >= 2:
            structure_score += 5
        score += structure_score

        # Keyword optimization assessment (25 points)
        optimal_keywords = [k for k in keyword_analysis if k.status == "optimal"]
        keyword_score = min(25, len(optimal_keywords) * 8)  # Max 25 points
        score += keyword_score

        # Determine quality level
        percentage = (score / max_score) * 100

        if percentage >= 85:
            return ContentQuality.EXCELLENT
        elif percentage >= 70:
            return ContentQuality.GOOD
        elif percentage >= 55:
            return ContentQuality.AVERAGE
        else:
            return ContentQuality.POOR

    def _calculate_seo_score(
        self,
        structure: ContentStructure,
        readability: ReadabilityScore,
        keyword_analysis: List[KeywordDensity],
        title: Optional[str] = None,
        meta_description: Optional[str] = None
    ) -> int:
        """Calculate comprehensive SEO score."""

        score = 0

        # Content length (20 points)
        if structure.word_count >= 1000:
            score += 20
        elif structure.word_count >= 500:
            score += 15
        elif structure.word_count >= 300:
            score += 10
        else:
            score += 5

        # Readability (20 points)
        if 60 <= readability.flesch_reading_ease <= 80:
            score += 20
        elif 50 <= readability.flesch_reading_ease <= 90:
            score += 15
        else:
            score += 10

        # Keyword optimization (20 points)
        optimal_keywords = [k for k in keyword_analysis if k.status == "optimal"]
        score += min(20, len(optimal_keywords) * 7)

        # Structure (20 points)
        if structure.heading_structure.get('h1', 0) >= 1:
            score += 5
        if sum(structure.heading_structure.values()) >= 3:
            score += 5
        if structure.paragraph_count >= 4:
            score += 5
        if structure.image_count >= 1:
            score += 5

        # Technical elements (20 points)
        if title and 30 <= len(title) <= 60:
            score += 10
        elif title:
            score += 5

        if meta_description and 120 <= len(meta_description) <= 160:
            score += 10
        elif meta_description:
            score += 5

        return min(100, score)

    def _generate_optimization_recommendations(
        self,
        structure: ContentStructure,
        readability: ReadabilityScore,
        keyword_analysis: List[KeywordDensity],
        quality: ContentQuality,
        title: Optional[str] = None,
        meta_description: Optional[str] = None
    ) -> List[str]:
        """Generate specific optimization recommendations."""

        recommendations = []

        # Content length recommendations
        if structure.word_count < self.target_word_count_min:
            recommendations.append(f"Expand content to at least {self.target_word_count_min} words for better SEO performance")
        elif structure.word_count > self.target_word_count_max:
            recommendations.append("Consider breaking this content into multiple focused articles")

        # Readability recommendations
        recommendations.extend(readability.recommendations)

        # Keyword recommendations
        low_density_keywords = [k for k in keyword_analysis if k.status == "low"]
        high_density_keywords = [k for k in keyword_analysis if k.status == "high"]

        if low_density_keywords:
            recommendations.append(f"Increase keyword density for: {', '.join([k.keyword for k in low_density_keywords])}")

        if high_density_keywords:
            recommendations.append(f"Reduce keyword density for: {', '.join([k.keyword for k in high_density_keywords])}")

        # Structure recommendations
        if structure.heading_structure.get('h1', 0) == 0:
            recommendations.append("Add an H1 heading to improve content structure")

        if sum(structure.heading_structure.values()) < 3:
            recommendations.append("Add more subheadings (H2, H3) to improve content organization")

        if structure.image_count == 0:
            recommendations.append("Add relevant images to enhance user engagement and SEO")

        if structure.link_count < 2:
            recommendations.append("Add more internal and external links to provide additional value")

        # Technical recommendations
        if not title:
            recommendations.append("Add a compelling title tag (30-60 characters)")
        elif len(title) < 30:
            recommendations.append("Expand title tag to 30-60 characters for better SEO")
        elif len(title) > 60:
            recommendations.append("Shorten title tag to under 60 characters")

        if not meta_description:
            recommendations.append("Add a meta description (120-160 characters)")
        elif len(meta_description) < 120:
            recommendations.append("Expand meta description to 120-160 characters")
        elif len(meta_description) > 160:
            recommendations.append("Shorten meta description to under 160 characters")

        return recommendations

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified approach)."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False

        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        # Ensure at least one syllable
        return max(1, syllable_count)

    def optimize_content_for_keywords(
        self,
        content: str,
        target_keywords: List[str],
        suggestions_only: bool = True
    ) -> Dict[str, Any]:
        """Provide content optimization suggestions for better keyword performance."""

        analysis = self.analyze_blog_content(content, target_keywords)

        optimization_suggestions = {
            'current_analysis': analysis,
            'optimization_opportunities': [],
            'content_suggestions': [],
            'technical_improvements': []
        }

        # Keyword optimization opportunities
        for keyword_data in analysis.keyword_analysis:
            if keyword_data.status == "low":
                optimization_suggestions['optimization_opportunities'].append({
                    'type': 'keyword_density',
                    'keyword': keyword_data.keyword,
                    'current_density': keyword_data.density,
                    'target_density': keyword_data.target_density,
                    'suggestion': f"Increase usage of '{keyword_data.keyword}' to reach optimal density"
                })

        # Content structure suggestions
        if analysis.content_structure.word_count < 500:
            optimization_suggestions['content_suggestions'].append({
                'type': 'content_length',
                'suggestion': 'Expand content with more detailed information, examples, and value-added sections',
                'target': 'Aim for 500-1500 words for better SEO performance'
            })

        if analysis.content_structure.heading_structure.get('h2', 0) < 2:
            optimization_suggestions['content_suggestions'].append({
                'type': 'content_structure',
                'suggestion': 'Add more H2 subheadings to improve content organization and readability',
                'target': 'Include 3-5 relevant subheadings'
            })

        # Technical improvements
        if analysis.readability.flesch_reading_ease < 60:
            optimization_suggestions['technical_improvements'].append({
                'type': 'readability',
                'suggestion': 'Improve readability by using shorter sentences and simpler words',
                'current_score': analysis.readability.flesch_reading_ease,
                'target_score': '60-80 (Standard to Easy reading level)'
            })

        return optimization_suggestions