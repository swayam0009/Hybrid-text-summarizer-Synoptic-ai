# utils/file_handler.py
import os
import io
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import PyPDF2
import docx
from bs4 import BeautifulSoup
import mimetypes

class FileHandler:
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.max_url_content_length = 5 * 1024 * 1024  # 5MB
        
        # Initialize supported extensions AFTER defining methods
        self.supported_extensions = {
            '.txt': self._read_text_file,
            '.pdf': self._read_pdf_file,
            '.docx': self._read_docx_file,
            '.doc': self._read_docx_file,
            '.html': self._read_html_file,
            '.htm': self._read_html_file
        }
    
    def _read_text_file(self, file_obj: io.BytesIO, encoding: str = 'utf-8') -> str:
        """Read plain text file"""
        try:
            return file_obj.read().decode(encoding)
        except UnicodeDecodeError:
            # Try different encodings
            for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    file_obj.seek(0)
                    return file_obj.read().decode(enc)
                except UnicodeDecodeError:
                    continue
            
            # If all fail, decode with errors='ignore'
            file_obj.seek(0)
            return file_obj.read().decode('utf-8', errors='ignore')
    
    def _read_pdf_file(self, file_obj: io.BytesIO, encoding: str = 'utf-8') -> str:
        """Read PDF file and extract text"""
        try:
            pdf_reader = PyPDF2.PdfReader(file_obj)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def _read_docx_file(self, file_obj: io.BytesIO, encoding: str = 'utf-8') -> str:
        """Read DOCX file and extract text"""
        try:
            doc = docx.Document(file_obj)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def _read_html_file(self, file_obj: io.BytesIO, encoding: str = 'utf-8') -> str:
        """Read HTML file and extract text"""
        try:
            html_content = file_obj.read().decode(encoding)
            return self._extract_text_from_html(html_content)
            
        except Exception as e:
            raise Exception(f"Error reading HTML: {str(e)}")
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract text from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def process_file(self, file_input: Union[str, io.BytesIO], 
                    file_type: Optional[str] = None,
                    encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Process various file types and extract text content
        """
        try:
            if isinstance(file_input, str):
                if file_input.startswith(('http://', 'https://')):
                    return self._process_url(file_input)
                else:
                    return self._process_file_path(file_input, encoding)
            else:
                return self._process_file_object(file_input, file_type, encoding)
                
        except Exception as e:
            return {
                'success': False,
                'error': f'File processing failed: {str(e)}',
                'text': '',
                'metadata': {}
            }
    
    def _process_file_path(self, file_path: str, encoding: str) -> Dict[str, Any]:
        """Process file from file path"""
        path = Path(file_path)
        
        if not path.exists():
            return {
                'success': False,
                'error': f'File not found: {file_path}',
                'text': '',
                'metadata': {}
            }
        
        if path.stat().st_size > self.max_file_size:
            return {
                'success': False,
                'error': f'File too large: {path.stat().st_size} bytes (max: {self.max_file_size})',
                'text': '',
                'metadata': {}
            }
        
        extension = path.suffix.lower()
        
        if extension not in self.supported_extensions:
            return {
                'success': False,
                'error': f'Unsupported file type: {extension}',
                'text': '',
                'metadata': {}
            }
        
        try:
            with open(path, 'rb') as file:
                text = self.supported_extensions[extension](file, encoding)
                
            return {
                'success': True,
                'text': text,
                'metadata': {
                    'filename': path.name,
                    'file_size': path.stat().st_size,
                    'file_type': extension,
                    'encoding': encoding
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error reading file: {str(e)}',
                'text': '',
                'metadata': {}
            }
    
    def _process_file_object(self, file_obj: io.BytesIO, 
                           file_type: Optional[str], 
                           encoding: str) -> Dict[str, Any]:
        """Process file from file-like object"""
        if file_type:
            extension = f'.{file_type.lower()}'
        else:
            extension = self._detect_file_type(file_obj)
        
        if extension not in self.supported_extensions:
            return {
                'success': False,
                'error': f'Unsupported file type: {extension}',
                'text': '',
                'metadata': {}
            }
        
        try:
            file_obj.seek(0)
            text = self.supported_extensions[extension](file_obj, encoding)
            
            return {
                'success': True,
                'text': text,
                'metadata': {
                    'file_type': extension,
                    'encoding': encoding
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing file: {str(e)}',
                'text': '',
                'metadata': {}
            }
    
    def _process_url(self, url: str) -> Dict[str, Any]:
        """Process content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content_length = len(response.content)
            if content_length > self.max_url_content_length:
                return {
                    'success': False,
                    'error': f'URL content too large: {content_length} bytes',
                    'text': '',
                    'metadata': {}
                }
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                text = self._extract_text_from_html(response.text)
            elif 'text/plain' in content_type:
                text = response.text
            elif 'application/pdf' in content_type:
                text = self._read_pdf_file(io.BytesIO(response.content))
            else:
                text = response.text
            
            return {
                'success': True,
                'text': text,
                'metadata': {
                    'url': url,
                    'content_type': content_type,
                    'content_length': content_length
                }
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Error fetching URL: {str(e)}',
                'text': '',
                'metadata': {}
            }
    
    def _detect_file_type(self, file_obj: io.BytesIO) -> str:
        """Detect file type from content"""
        file_obj.seek(0)
        header = file_obj.read(8)
        file_obj.seek(0)
        
        if header.startswith(b'%PDF'):
            return '.pdf'
        
        if header.startswith(b'PK\x03\x04'):
            return '.docx'
        
        try:
            file_obj.seek(0)
            content = file_obj.read(1024).decode('utf-8')
            if content.strip().startswith('<!DOCTYPE') or content.strip().startswith('<html'):
                return '.html'
            return '.txt'
        except:
            return '.txt'
    
    def validate_file_size(self, file_size: int) -> bool:
        """Validate file size against limits"""
        return file_size <= self.max_file_size
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return list(self.supported_extensions.keys())
    
    def estimate_processing_time(self, file_size: int, file_type: str) -> float:
        """Estimate processing time in seconds"""
        base_times = {
            '.txt': 0.1,
            '.pdf': 2.0,
            '.docx': 1.0,
            '.html': 0.5
        }
        
        size_mb = file_size / (1024 * 1024)
        base_time = base_times.get(file_type, 1.0)
        
        return size_mb * base_time
