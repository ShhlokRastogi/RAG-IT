import fitz  # PyMuPDF
# Suppress non-fatal PDF structure tree warnings from showing in the console
fitz.TOOLS.mupdf_display_errors = False
fitz.TOOLS.mupdf_display_warnings = False
import base64
from pathlib import Path
import os
import uuid
from multimodal_rag.config import MEDIA_DIR, get_openai_api_key, OPENAI_VLM_MODEL
from openai import OpenAI

def encode_image_to_base64(image_path: Path) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def bbox_intersects_any(rect: fitz.Rect, bboxes: list[fitz.Rect], threshold=0.6) -> bool:
    """Check if a rect mostly overlaps with any bounding box in the list."""
    for bbox in bboxes:
        intersect = rect & bbox
        if intersect.is_empty:
            continue
        # If intersection area occupies a significant part of the rect area
        rect_area = rect.get_area()
        if rect_area > 0 and (intersect.get_area() / rect_area) > threshold:
            return True
    return False

def list_to_markdown_table(rows: list[list[str]]) -> str:
    """Convert list of rows into standard markdown table."""
    if not rows or not rows[0]:
        return ""
    headers = [str(h).strip().replace("\n", " ") for h in rows[0]]
    # Ensure unique and non-empty headers
    headers = [h if h else f"Col_{i}" for i, h in enumerate(headers)]
    
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for row in rows[1:]:
        cells = [str(c).strip().replace("\n", " ") if c is not None else "" for c in row]
        # Align length
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        else:
            cells = cells[:len(headers)]
        md += "| " + " | ".join(cells) + " |\n"
    return md

class DocumentProcessor:
    """Parses PDF documents into layout-aware smart chunks with visual features."""
    def __init__(self):
        self.api_key = get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def reload_client(self):
        """Reload API key and client from config."""
        self.api_key = get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def process_pdf(self, pdf_path: str) -> list[dict]:
        """
        Parses a PDF into smart chunks: text, tables, and images.
        Returns a list of dicts, where each dict represents a chunk:
        {
           "id": str,
           "document_name": str,
           "page_number": int,
           "section_title": str,
           "type": "text" | "table" | "image",
           "content": str,         # Text content, Markdown table, or Image caption/description
           "image_path": str,      # Path to cropped image if table/image type
           "metadata": dict        # Page number, document name, visual tags
        }
        """
        self.reload_client()
        pdf_path = Path(pdf_path)
        doc_name = pdf_path.name
        chunks = []
        
        doc = fitz.open(pdf_path)
        
        # 1. Calculate document-wide average font size for header detection
        font_sizes = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            size = s.get("size", 10)
                            font_sizes.append(size)
                            
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10.0
        header_threshold = avg_font_size * 1.2  # 20% larger than average is considered header
        
        current_section = "Introduction"
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"[Parser] Processing page {page_num + 1}/{len(doc)} of {doc_name}...")
            
            # --- A. Table Bounding Boxes ---
            tables = page.find_tables()
            table_bboxes = [fitz.Rect(t.bbox) for t in tables]
            
            # --- B. Image Bounding Boxes ---
            images_info = page.get_image_info(xrefs=True)
            image_bboxes = [fitz.Rect(img["bbox"]) for img in images_info if img["bbox"][2] - img["bbox"][0] > 10 and img["bbox"][3] - img["bbox"][1] > 10]
            
            # Merge bboxes for overlapping tables/images to prevent duplicate cropping
            # We treat them separately for classification
            
            # --- C. Crop and Parse Tables ---
            for idx, table in enumerate(tables):
                t_bbox = fitz.Rect(table.bbox)
                # Crop table as image
                table_img_filename = f"table_{uuid.uuid4().hex[:8]}_p{page_num+1}_{idx}.png"
                table_img_path = MEDIA_DIR / table_img_filename
                
                # Render cropped table page section
                pix = page.get_pixmap(clip=t_bbox, dpi=150)
                pix.save(table_img_path)
                
                # Extract structured rows
                raw_rows = table.extract()
                raw_table_md = list_to_markdown_table(raw_rows)
                
                table_content = ""
                # Use VLM if client is available
                if self.client:
                    try:
                        base64_img = encode_image_to_base64(table_img_path)
                        response = self.client.chat.completions.create(
                            model=OPENAI_VLM_MODEL,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text", 
                                            "text": (
                                                "Convert this table image into clean Markdown format. "
                                                "Preserve headers, rows, and relationships accurately. "
                                                "Below the markdown table, write a brief 2-3 sentence summary of the "
                                                "data, trends, or comparisons. If the image is cut off or not a table, write your best interpretation."
                                                f"\n\nDraft extracted table data:\n{raw_table_md}"
                                            )
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                                        }
                                    ]
                                }
                            ]
                        )
                        table_content = response.choices[0].message.content
                        if hasattr(response, 'usage') and response.usage:
                            from multimodal_rag.config import TokenTracker
                            TokenTracker.add_chat_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
                    except Exception as e:
                        print(f"[Parser] OpenAI table parsing failed: {e}. Falling back to local extract.")
                        table_content = f"{raw_table_md}\n\n[Summary: Table on page {page_num+1}. Local table structure extraction.]"
                else:
                    table_content = f"{raw_table_md}\n\n[Summary: Table on page {page_num+1}. Local table structure extraction.]"
                
                chunk_id = f"table_{doc_name.replace(' ', '_')}_p{page_num+1}_{idx}"
                chunks.append({
                    "id": chunk_id,
                    "document_name": doc_name,
                    "page_number": page_num + 1,
                    "section_title": current_section,
                    "type": "table",
                    "content": table_content,
                    "image_path": str(table_img_path),
                    "metadata": {
                        "document_name": doc_name,
                        "page_number": page_num + 1,
                        "section_title": current_section,
                        "type": "table",
                        "image_path": table_img_filename,
                        "table_id": chunk_id
                    }
                })
            
            # --- D. Crop and Parse Images ---
            for idx, img_info in enumerate(images_info):
                i_bbox = fitz.Rect(img_info["bbox"])
                # Avoid tiny images or images overlapping with tables
                if i_bbox.get_area() < 100 or bbox_intersects_any(i_bbox, table_bboxes):
                    continue
                
                img_filename = f"image_{uuid.uuid4().hex[:8]}_p{page_num+1}_{idx}.png"
                img_path = MEDIA_DIR / img_filename
                
                # Render cropped image section
                pix = page.get_pixmap(clip=i_bbox, dpi=150)
                pix.save(img_path)
                
                image_description = ""
                # Use VLM if client is available
                if self.client:
                    try:
                        base64_img = encode_image_to_base64(img_path)
                        response = self.client.chat.completions.create(
                            model=OPENAI_VLM_MODEL,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "Analyze this image from a document. "
                                                "If it is a diagram, flowchart, chart, screenshot, or graphic, explain it in detail: "
                                                "what it represents, labels, trends, values, and logic. "
                                                "Extract all visible text as OCR output and present it at the end under 'OCR Text:'."
                                            )
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                                        }
                                    ]
                                }
                            ]
                        )
                        image_description = response.choices[0].message.content
                        if hasattr(response, 'usage') and response.usage:
                            from multimodal_rag.config import TokenTracker
                            TokenTracker.add_chat_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
                    except Exception as e:
                        print(f"[Parser] OpenAI image parsing failed: {e}. Falling back to default caption.")
                        image_description = f"[Diagram/Figure on page {page_num+1}. Bounding Box: {img_info['bbox']}. Set OpenAI Key for description.]"
                else:
                    image_description = f"[Diagram/Figure on page {page_num+1}. Bounding Box: {img_info['bbox']}. Set OpenAI Key for description.]"
                
                chunk_id = f"image_{doc_name.replace(' ', '_')}_p{page_num+1}_{idx}"
                chunks.append({
                    "id": chunk_id,
                    "document_name": doc_name,
                    "page_number": page_num + 1,
                    "section_title": current_section,
                    "type": "image",
                    "content": image_description,
                    "image_path": str(img_path),
                    "metadata": {
                        "document_name": doc_name,
                        "page_number": page_num + 1,
                        "section_title": current_section,
                        "type": "image",
                        "image_path": img_filename,
                        "image_id": chunk_id
                    }
                })
            
            # --- E. Extract and Layout-Filter Text ---
            blocks = page.get_text("dict")["blocks"]
            text_blocks_to_process = []
            
            for b in blocks:
                if "lines" not in b:
                    continue
                b_rect = fitz.Rect(b["bbox"])
                
                # Check if this text block is mostly inside any table or image bbox
                if bbox_intersects_any(b_rect, table_bboxes) or bbox_intersects_any(b_rect, image_bboxes):
                    continue
                
                # Collect text lines
                block_text = []
                is_header = False
                
                for l in b["lines"]:
                    for s in l["spans"]:
                        span_text = s["text"].strip()
                        if not span_text:
                            continue
                        
                        # Detect section headers
                        span_size = s.get("size", 10.0)
                        span_flags = s.get("flags", 0)
                        
                        # Font size threshold or bold (flags & 2 is bold in PyMuPDF)
                        if span_size > header_threshold or (span_size > avg_font_size * 1.1 and (span_flags & 2)):
                            is_header = True
                            current_section = span_text
                        
                        block_text.append(span_text)
                
                block_str = " ".join(block_text).strip()
                if block_str:
                    text_blocks_to_process.append({
                        "text": block_str,
                        "is_header": is_header,
                        "bbox": b["bbox"]
                    })
            
            # --- F. Smart Text Chunking ---
            current_chunk_text = []
            current_chunk_len = 0
            
            for block in text_blocks_to_process:
                # If block is a header, dump the current chunk first
                if block["is_header"] and current_chunk_text:
                    chunk_content = " ".join(current_chunk_text).strip()
                    chunk_id = f"text_{doc_name.replace(' ', '_')}_p{page_num+1}_{len(chunks)}"
                    chunks.append({
                        "id": chunk_id,
                        "document_name": doc_name,
                        "page_number": page_num + 1,
                        "section_title": current_section,
                        "type": "text",
                        "content": chunk_content,
                        "image_path": "",
                        "metadata": {
                            "document_name": doc_name,
                            "page_number": page_num + 1,
                            "section_title": current_section,
                            "type": "text"
                        }
                    })
                    current_chunk_text = []
                    current_chunk_len = 0
                
                current_chunk_text.append(block["text"])
                current_chunk_len += len(block["text"])
                
                # Break chunk if it exceeds limit (approx 800 characters)
                if current_chunk_len >= 800:
                    chunk_content = " ".join(current_chunk_text).strip()
                    chunk_id = f"text_{doc_name.replace(' ', '_')}_p{page_num+1}_{len(chunks)}"
                    chunks.append({
                        "id": chunk_id,
                        "document_name": doc_name,
                        "page_number": page_num + 1,
                        "section_title": current_section,
                        "type": "text",
                        "content": chunk_content,
                        "image_path": "",
                        "metadata": {
                            "document_name": doc_name,
                            "page_number": page_num + 1,
                            "section_title": current_section,
                            "type": "text"
                        }
                    })
                    current_chunk_text = []
                    current_chunk_len = 0
            
            # Clear remaining text blocks on the page
            if current_chunk_text:
                chunk_content = " ".join(current_chunk_text).strip()
                chunk_id = f"text_{doc_name.replace(' ', '_')}_p{page_num+1}_{len(chunks)}"
                chunks.append({
                    "id": chunk_id,
                    "document_name": doc_name,
                    "page_number": page_num + 1,
                    "section_title": current_section,
                    "type": "text",
                    "content": chunk_content,
                    "image_path": "",
                    "metadata": {
                        "document_name": doc_name,
                        "page_number": page_num + 1,
                        "section_title": current_section,
                        "type": "text"
                    }
                })
                
        doc.close()
        print(f"[Parser] Extraction complete! Created {len(chunks)} chunks from {doc_name}.")
        return chunks

    def process_file(self, file_path: str) -> list[dict]:
        """Parses any supported file (PDF, Excel, Word, CSV, Text, Markdown) into standard chunks."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            return self.process_pdf(str(file_path))
        elif suffix in [".xlsx", ".xls"]:
            return self.process_excel(str(file_path))
        elif suffix == ".csv":
            return self.process_csv(str(file_path))
        elif suffix == ".docx":
            return self.process_docx(str(file_path))
        elif suffix in [".txt", ".md"]:
            return self.process_text_md(str(file_path))
        else:
            print(f"[Parser] Unsupported file extension: {suffix}")
            return []

    def process_csv(self, file_path: str) -> list[dict]:
        """Parses CSV sheets into structured Markdown table chunks, splitting large tables."""
        import pandas as pd
        file_path = Path(file_path)
        doc_name = file_path.name
        chunks = []
        
        try:
            df = pd.read_csv(file_path)
            df = df.fillna("")
            
            chunk_size = 50
            total_rows = len(df)
            
            if total_rows <= chunk_size:
                markdown_table = df.to_markdown(index=False)
                chunk_id = f"table_{doc_name.replace(' ', '_')}"
                content = f"Table: {doc_name}\n\n{markdown_table}"
                chunks.append({
                    "id": chunk_id,
                    "document_name": doc_name,
                    "page_number": 1,
                    "section_title": doc_name,
                    "type": "table",
                    "content": content,
                    "image_path": "",
                    "metadata": {
                        "document_name": doc_name,
                        "page_number": 1,
                        "section_title": doc_name,
                        "type": "table",
                        "image_path": "",
                        "table_id": chunk_id
                    }
                })
            else:
                for i in range(0, total_rows, chunk_size):
                    sub_df = df.iloc[i : i + chunk_size]
                    markdown_table = sub_df.to_markdown(index=False)
                    part_num = (i // chunk_size) + 1
                    total_parts = (total_rows + chunk_size - 1) // chunk_size
                    
                    chunk_id = f"table_{doc_name.replace(' ', '_')}_part_{part_num}"
                    content = f"Table: {doc_name} (Part {part_num} of {total_parts}, Rows {i+1} to {min(i+chunk_size, total_rows)})\n\n{markdown_table}"
                    
                    chunks.append({
                        "id": chunk_id,
                        "document_name": doc_name,
                        "page_number": part_num,
                        "section_title": f"{doc_name} Part {part_num}",
                        "type": "table",
                        "content": content,
                        "image_path": "",
                        "metadata": {
                            "document_name": doc_name,
                            "page_number": part_num,
                            "section_title": f"{doc_name} Part {part_num}",
                            "type": "table",
                            "image_path": "",
                            "table_id": chunk_id
                        }
                    })
        except Exception as e:
            print(f"[Parser] CSV parsing failed: {e}")
        return chunks

    def process_excel(self, file_path: str) -> list[dict]:
        """Parses Excel sheets into structured Markdown table chunks, splitting large sheets."""
        import pandas as pd
        file_path = Path(file_path)
        doc_name = file_path.name
        chunks = []
        
        try:
            excel_file = pd.read_excel(file_path, sheet_name=None)
            for idx, (sheet_name, df) in enumerate(excel_file.items()):
                df = df.fillna("")
                chunk_size = 50
                total_rows = len(df)
                
                if total_rows <= chunk_size:
                    markdown_table = df.to_markdown(index=False)
                    chunk_id = f"table_{doc_name.replace(' ', '_')}_sheet_{idx}"
                    content = f"Sheet: {sheet_name}\n\n{markdown_table}"
                    chunks.append({
                        "id": chunk_id,
                        "document_name": doc_name,
                        "page_number": idx + 1,
                        "section_title": f"Sheet: {sheet_name}",
                        "type": "table",
                        "content": content,
                        "image_path": "",
                        "metadata": {
                            "document_name": doc_name,
                            "page_number": idx + 1,
                            "section_title": f"Sheet: {sheet_name}",
                            "type": "table",
                            "image_path": "",
                            "table_id": chunk_id
                        }
                    })
                else:
                    for i in range(0, total_rows, chunk_size):
                        sub_df = df.iloc[i : i + chunk_size]
                        markdown_table = sub_df.to_markdown(index=False)
                        part_num = (i // chunk_size) + 1
                        total_parts = (total_rows + chunk_size - 1) // chunk_size
                        
                        chunk_id = f"table_{doc_name.replace(' ', '_')}_sheet_{idx}_part_{part_num}"
                        content = f"Sheet: {sheet_name} (Part {part_num} of {total_parts}, Rows {i+1} to {min(i+chunk_size, total_rows)})\n\n{markdown_table}"
                        
                        virtual_page = (idx + 1) * 100 + part_num
                        
                        chunks.append({
                            "id": chunk_id,
                            "document_name": doc_name,
                            "page_number": virtual_page,
                            "section_title": f"Sheet: {sheet_name} Part {part_num}",
                            "type": "table",
                            "content": content,
                            "image_path": "",
                            "metadata": {
                                "document_name": doc_name,
                                "page_number": virtual_page,
                                "section_title": f"Sheet: {sheet_name} Part {part_num}",
                                "type": "table",
                                "image_path": "",
                                "table_id": chunk_id
                            }
                        })
        except Exception as e:
            print(f"[Parser] Excel parsing failed: {e}")
        return chunks

    def process_docx(self, file_path: str) -> list[dict]:
        """Parses Word docx block paragraphs and tables in sequential relative order."""
        from docx import Document
        file_path = Path(file_path)
        doc_name = file_path.name
        chunks = []
        
        try:
            doc = Document(file_path)
            current_section = "Introduction"
            chunk_idx = 0
            text_accumulator = []
            
            def flush_text():
                nonlocal chunk_idx
                if text_accumulator:
                    content = "\n".join(text_accumulator).strip()
                    if content:
                        chunk_id = f"text_{doc_name.replace(' ', '_')}_chunk_{chunk_idx}"
                        chunks.append({
                            "id": chunk_id,
                            "document_name": doc_name,
                            "page_number": 1,
                            "section_title": current_section,
                            "type": "text",
                            "content": content,
                            "image_path": "",
                            "metadata": {
                                "document_name": doc_name,
                                "page_number": 1,
                                "section_title": current_section,
                                "type": "text",
                                "image_path": ""
                            }
                        })
                        chunk_idx += 1
                    text_accumulator.clear()
            
            for element in doc.element.body:
                if element.tag.endswith('p'):
                    from docx.text.paragraph import Paragraph
                    p = Paragraph(element, doc)
                    text = p.text.strip()
                    if not text:
                        continue
                        
                    if p.style.name.startswith("Heading") or (len(text) < 100 and p.style.name == "Title"):
                        flush_text()
                        current_section = text
                        text_accumulator.append(f"## {text}")
                    else:
                        text_accumulator.append(text)
                        if sum(len(t) for t in text_accumulator) > 1000:
                            flush_text()
                            
                elif element.tag.endswith('tbl'):
                    flush_text()
                    from docx.table import Table
                    tbl = Table(element, doc)
                    
                    rows = []
                    for row in tbl.rows:
                        rows.append([cell.text.strip() for cell in row.cells])
                    
                    markdown_table = list_to_markdown_table(rows)
                    chunk_id = f"table_{doc_name.replace(' ', '_')}_chunk_{chunk_idx}"
                    chunks.append({
                        "id": chunk_id,
                        "document_name": doc_name,
                        "page_number": 1,
                        "section_title": current_section,
                        "type": "table",
                        "content": markdown_table,
                        "image_path": "",
                        "metadata": {
                            "document_name": doc_name,
                            "page_number": 1,
                            "section_title": current_section,
                            "type": "table",
                            "image_path": "",
                            "table_id": chunk_id
                        }
                    })
                    chunk_idx += 1
            
            flush_text()
        except Exception as e:
            print(f"[Parser] Word document parsing failed: {e}")
        return chunks

    def process_text_md(self, file_path: str) -> list[dict]:
        """Parses text and markdown documents, segmenting by header indicators."""
        file_path = Path(file_path)
        doc_name = file_path.name
        chunks = []
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
                
            lines = text.split("\n")
            current_section = "Main"
            text_accumulator = []
            chunk_idx = 0
            
            def flush_text():
                nonlocal chunk_idx
                if text_accumulator:
                    content = "\n".join(text_accumulator).strip()
                    if content:
                        chunk_id = f"text_{doc_name.replace(' ', '_')}_chunk_{chunk_idx}"
                        chunks.append({
                            "id": chunk_id,
                            "document_name": doc_name,
                            "page_number": 1,
                            "section_title": current_section,
                            "type": "text",
                            "content": content,
                            "image_path": "",
                            "metadata": {
                                "document_name": doc_name,
                                "page_number": 1,
                                "section_title": current_section,
                                "type": "text",
                                "image_path": ""
                            }
                        })
                        chunk_idx += 1
                    text_accumulator.clear()
                    
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#"):
                    flush_text()
                    current_section = stripped.lstrip("#").strip()
                    text_accumulator.append(line)
                else:
                    text_accumulator.append(line)
                    if sum(len(t) for t in text_accumulator) > 1000:
                        flush_text()
            flush_text()
        except Exception as e:
            print(f"[Parser] Text/Markdown parsing failed: {e}")
        return chunks
