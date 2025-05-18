  # --- Python/FastAPI Backend Code (main.py) ---
import io
import re
import logging
import traceback
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import fitz  # PyMuPDF
import json
from fpdf import FPDF
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Configure logging - Set to DEBUG for detailed extraction info
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Marks Summarizer API (PyMuPDF)", version="1.5.1") # Incremented version for fix

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Global exception handler middleware
@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except HTTPException as he:
        logger.warning(f"HTTP Exception caught: Status={he.status_code}, Detail={he.detail}")
        raise he # Re-raise to let FastAPI handle it or other middleware
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled exception for request {request.method} {request.url.path}: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred processing the request.", "error": str(e)},
        )

# --------------------------------------------------------------------------
# Data Extraction Function using PyMuPDF (fitz)
# --------------------------------------------------------------------------
def extract_marks_fitz(pdf_content: bytes):
    logger.info("Starting PDF extraction process using PyMuPDF (fitz).")
    all_students = []
    processed_usns = set()
    doc = None
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        logger.info(f"PDF opened successfully with fitz. Pages: {doc.page_count}")
        # Regex improved slightly to be more robust with spacing around (TH) and (PR)
        pattern = r'(\d+)\s+([A-Z0-9]{10})\s+(?:[A-Z\s.]+)?\s*(\d+)\s*\(\s*(?:TH|Theory)\s*\)\s*[,;]?\s*(\d+)\s*\(\s*(?:PR|Practical|PRA|PRAC)\s*\)'

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            page_id = f"Page {page_num + 1}"
            if not text: logger.warning(f"{page_id}: No text extracted."); continue
            logger.debug(f"{page_id}: Text length: {len(text)}")

            matches = re.finditer(pattern, text, re.IGNORECASE)
            found_on_page = 0
            for match_num, match in enumerate(matches, 1):
                try:
                    usn = match.group(2).strip().upper()
                    if usn in processed_usns: continue # Skip duplicates

                    sl_no_val = int(match.group(1))
                    theory_val = int(match.group(3))
                    practical_val = int(match.group(4))

                    if not (0 <= theory_val <= 100 and 0 <= practical_val <= 100):
                        logger.warning(f"{page_id} Match {match_num}: USN={usn} - Invalid marks (TH={theory_val}, PR={practical_val}). Skipping.")
                        continue

                    student = {'SL_NO': sl_no_val, 'USN': usn, 'Theory': theory_val, 'Practical': practical_val}
                    all_students.append(student)
                    processed_usns.add(usn)
                    found_on_page += 1
                    logger.debug(f"{page_id} Match {match_num}: Added USN={usn}, TH={theory_val}, PR={practical_val}")

                except Exception as match_err:
                    logger.warning(f"{page_id} Match {match_num}: Error processing match: {match_err}. Groups: {match.groups()}")

            if found_on_page > 0: logger.info(f"{page_id}: Found {found_on_page} valid records.")
            else: logger.info(f"{page_id}: No records matched pattern.")

        if not all_students: raise ValueError("No valid student records extracted from the PDF.")
        logger.info(f"Successfully extracted {len(all_students)} unique records using PyMuPDF.")
        return pd.DataFrame(all_students)

    except ValueError as ve: raise HTTPException(status_code=400, detail=str(ve))
    except fitz.fitz.FileDataError as fitz_err: raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file provided.")
    except Exception as e: logger.error(f"PDF processing error (PyMuPDF): {e}", exc_info=True); raise HTTPException(status_code=500, detail="Server error during PDF processing.")
    finally:
        if doc: doc.close(); logger.debug("PyMuPDF doc closed.")


# --------------------------------------------------------------------------
# Data Analysis Function
# --------------------------------------------------------------------------
def analyze_marks(df: pd.DataFrame):
    if df.empty:
        logger.error("Analysis error: Input DataFrame is empty.")
        raise ValueError("Cannot analyze empty dataset.")

    logger.info(f"Starting marks analysis for {len(df)} records.")
    try:
        THEORY_PASS_THRESHOLD = 10
        PRACTICAL_PASS_THRESHOLD = 10

        if 'Theory' not in df.columns or 'Practical' not in df.columns: raise KeyError("Missing 'Theory' or 'Practical' columns.")
        df['Theory'] = pd.to_numeric(df['Theory'], errors='coerce')
        df['Practical'] = pd.to_numeric(df['Practical'], errors='coerce')

        initial_rows = len(df)
        df = df.dropna(subset=['Theory', 'Practical'])
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0: logger.warning(f"Dropped {dropped_rows} rows due to non-numeric marks.")

        if df.empty: raise ValueError("No valid numeric marks data remained after cleaning.")

        df['Theory_Pass'] = df['Theory'] >= THEORY_PASS_THRESHOLD
        df['Practical_Pass'] = df['Practical'] >= PRACTICAL_PASS_THRESHOLD
        df['Overall_Pass'] = df['Theory_Pass'] & df['Practical_Pass']
        df['Total'] = df['Theory'] + df['Practical']
        df['Theory_Eligible'] = df['Theory_Pass'].map({True: 'Yes', False: 'No'})
        df['Practical_Eligible'] = df['Practical_Pass'].map({True: 'Yes', False: 'No'})
        df['Overall_Eligible'] = df['Overall_Pass'].map({True: 'Yes', False: 'No'})

        total_students = len(df)
        passed_count = int(df['Overall_Pass'].sum())
        failed_count = total_students - passed_count
        pass_percentage = round((passed_count / total_students) * 100, 2) if total_students > 0 else 0.0
        theory_passed_count = int(df['Theory_Pass'].sum())
        practical_passed_count = int(df['Practical_Pass'].sum())

        theory_stats = { 'average': round(df['Theory'].mean(), 2), 'max': df['Theory'].max().item(), 'min': df['Theory'].min().item(), 'passed': theory_passed_count, 'pass_percentage': round((theory_passed_count / total_students) * 100, 2) if total_students > 0 else 0.0 }
        practical_stats = { 'average': round(df['Practical'].mean(), 2), 'max': df['Practical'].max().item(), 'min': df['Practical'].min().item(), 'passed': practical_passed_count, 'pass_percentage': round((practical_passed_count / total_students) * 100, 2) if total_students > 0 else 0.0 }

        student_list_cols = ['SL_NO', 'USN', 'Theory', 'Theory_Eligible', 'Practical', 'Practical_Eligible', 'Total', 'Overall_Eligible']
        cols_to_select = [col for col in student_list_cols if col in df.columns]
        logger.debug(f"Columns selected for student lists: {cols_to_select}")


        summary = {
            'processing_summary': { 'total_records_processed': total_students, 'pass_threshold_theory': THEORY_PASS_THRESHOLD, 'pass_threshold_practical': PRACTICAL_PASS_THRESHOLD, },
            'overall_summary': { 'passed_count': passed_count, 'failed_count': failed_count, 'pass_percentage': pass_percentage, },
            'theory_stats': theory_stats,
            'practical_stats': practical_stats,
            'top_students': df.nlargest(5, 'Total')[cols_to_select].to_dict('records'),
            'bottom_students': df.nsmallest(5, 'Total')[cols_to_select].to_dict('records'),
            'all_students': df[cols_to_select].to_dict('records')
        }
        logger.debug(f"Generated list lengths: top={len(summary['top_students'])}, bottom={len(summary['bottom_students'])}, all={len(summary['all_students'])}")

        logger.info("Marks analysis completed successfully.")
        return summary

    except KeyError as ke: logger.error(f"Analysis KeyError: {ke}", exc_info=True); raise ValueError(f"Internal analysis error: Missing column '{ke}'.")
    except ValueError as ve: logger.error(f"Analysis ValueError: {ve}", exc_info=True); raise
    except Exception as e: logger.error(f"Analysis Unexpected Error: {e}", exc_info=True); raise ValueError(f"Unexpected analysis error.")

# --------------------------------------------------------------------------
# JSON Serializer Helper
# --------------------------------------------------------------------------
def default_serializer(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj): return None # Handle NaN/Inf
        return float(obj)
    elif isinstance(obj, (np.ndarray,)): return obj.tolist()
    elif pd.isna(obj): return None # Explicitly handle pandas NaT or NaN
    return str(obj)

# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------
@app.post("/summarize-marks/", tags=["Marks Processing"])
async def summarize_marks_endpoint(file: UploadFile = File(..., description="PDF marks file.")): # Renamed for clarity
    if not file.filename.lower().endswith('.pdf'): raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    logger.info(f"Received file: {file.filename}")
    pdf_content = None
    try:
        pdf_content = await file.read()
        if not pdf_content: raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        MAX_FILE_SIZE = 20 * 1024 * 1024 # 20 MB
        if len(pdf_content) > MAX_FILE_SIZE: raise HTTPException(status_code=413, detail=f"File size exceeds limit of {MAX_FILE_SIZE // (1024*1024)}MB.")
        
        df_marks = extract_marks_fitz(pdf_content)
        analysis_result_dict = analyze_marks(df_marks)
        
        json_compatible_string = json.dumps(analysis_result_dict, default=default_serializer)
        return JSONResponse(content=json.loads(json_compatible_string))

    except (ValueError, KeyError) as processing_err: 
        logger.error(f"Processing Error: {processing_err}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(processing_err)}")
    except HTTPException as he: 
        raise he
    except Exception as e: 
        logger.error(f"Endpoint Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")
    finally:
        if file: await file.close()
        # pdf_content = None # Not strictly necessary here as it's local to function scope


@app.get("/", tags=["General"])
async def root(): return {"message": "Marks Summarizer API (PyMuPDF). Use /docs."}

@app.get("/health", tags=["General"])
async def health_check(): return {"status": "healthy"}


# --- PDF Generation Logic ---
class PDFReport(FPDF):
    EFFECTIVE_PAGE_WIDTH = 190  # A4 width 210mm - 10mm margin each side

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Marks Eligibility Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230) # Light gray background
        self.cell(self.EFFECTIVE_PAGE_WIDTH, 8, title, 0, 1, 'L', fill=True) # Changed width to fill
        self.ln(3)

    def stat_item(self, label, value):
        self.set_font('Arial', 'B', 10)
        self.cell(self.EFFECTIVE_PAGE_WIDTH * 0.4, 7, str(label), 0, 0, 'L') # Ensure label is string
        self.set_font('Arial', '', 10)
        self.cell(self.EFFECTIVE_PAGE_WIDTH * 0.6, 7, str(value), 0, 1, 'L') # Ensure value is string

    def student_table(self, title, headers_map: Dict[str, str], data: List[Dict], column_widths_percent: List[float]):
        if not data:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f"{title}: No data available.", 0, 1)
            self.ln(5)
            return

        self.section_title(title)
        self.set_font('Arial', 'B', 9)
        
        actual_headers = list(headers_map.values())
        data_keys = list(headers_map.keys())

        column_widths = [self.EFFECTIVE_PAGE_WIDTH * (p / 100.0) for p in column_widths_percent]

        for i, header_text in enumerate(actual_headers):
            self.cell(column_widths[i], 7, header_text, 1, 0, 'C')
        self.ln()

        self.set_font('Arial', '', 8)
        for row_data in data:
            for i, data_key in enumerate(data_keys):
                cell_text = str(row_data.get(data_key, 'N/A')) # Ensure text is string and handle missing keys
                # Truncate or wrap long text if necessary (simple example, more advanced needed for true wrapping)
                max_len = int(column_widths[i] / 1.8) # Approx char width, adjust as needed
                if len(cell_text) > max_len and max_len > 3:
                     cell_text = cell_text[:max_len-3] + "..."

                self.cell(column_widths[i], 6, cell_text, 1, 0, 'L')
            self.ln()
        self.ln(5)

# --- Pydantic Models for PDF Export Request Body ---
class ProcessingSummaryModel(BaseModel):
    total_records_processed: Optional[int] = None
    pass_threshold_theory: Optional[int] = None
    pass_threshold_practical: Optional[int] = None

class OverallSummaryModel(BaseModel):
    passed_count: Optional[int] = None
    failed_count: Optional[int] = None
    pass_percentage: Optional[float] = None

class StatsModel(BaseModel):
    average: Optional[float] = None
    max: Optional[Any] = None
    min: Optional[Any] = None
    passed: Optional[int] = None
    pass_percentage: Optional[float] = None

class StudentModel(BaseModel):
    SL_NO: Optional[int] = None
    USN: Optional[str] = None
    Theory: Optional[Any] = None
    Practical: Optional[Any] = None
    Total: Optional[Any] = None
    Overall_Eligible: Optional[str] = None
    Theory_Eligible: Optional[str] = None
    Practical_Eligible: Optional[str] = None


class MarksDataForPDF(BaseModel):
    processing_summary: ProcessingSummaryModel
    overall_summary: OverallSummaryModel
    theory_stats: StatsModel
    practical_stats: StatsModel
    top_students: List[StudentModel]
    bottom_students: List[StudentModel]
    all_students: List[StudentModel]


@app.post("/export-pdf/", tags=["Exporting"], response_class=StreamingResponse)
async def export_results_as_pdf(data: MarksDataForPDF = Body(...)):
    logger.info("Received request to export data as PDF.")
    try:
        pdf = PDFReport()
        pdf.add_page()

        pdf.section_title("Overall Statistics")
        pdf.stat_item("Total Students Processed", data.processing_summary.total_records_processed)
        pdf.stat_item("Eligible Students", data.overall_summary.passed_count)
        pdf.stat_item("Not Eligible Students", data.overall_summary.failed_count)
        pdf.stat_item("Pass Percentage", f"{data.overall_summary.pass_percentage}%" if data.overall_summary.pass_percentage is not None else "N/A")
        pdf.ln(5)

        pdf.section_title("Theory Marks Statistics")
        pdf.stat_item("Pass Threshold", data.processing_summary.pass_threshold_theory)
        pdf.stat_item("Average", data.theory_stats.average if data.theory_stats.average is not None else "N/A")
        pdf.stat_item("Highest", data.theory_stats.max)
        pdf.stat_item("Lowest", data.theory_stats.min)
        pdf.stat_item("Passed", data.theory_stats.passed)
        pdf.ln(5)

        pdf.section_title("Practical Marks Statistics")
        pdf.stat_item("Pass Threshold", data.processing_summary.pass_threshold_practical)
        pdf.stat_item("Average", data.practical_stats.average if data.practical_stats.average is not None else "N/A")
        pdf.stat_item("Highest", data.practical_stats.max)
        pdf.stat_item("Lowest", data.practical_stats.min)
        pdf.stat_item("Passed", data.practical_stats.passed)
        pdf.ln(5)

        top_students_headers = {"USN": "USN", "Theory": "Theory", "Practical": "Practical", "Total": "Total", "Overall_Eligible": "Eligibility"}
        top_students_widths = [25, 15, 15, 15, 30]
        pdf.student_table("Top 5 Students", top_students_headers, [s.model_dump(exclude_none=True) for s in data.top_students], top_students_widths)

        bottom_students_headers = {"USN": "USN", "Theory": "Theory", "Practical": "Practical", "Total": "Total", "Overall_Eligible": "Eligibility"}
        bottom_students_widths = [25, 15, 15, 15, 30]
        pdf.student_table("Bottom 5 Students", bottom_students_headers, [s.model_dump(exclude_none=True) for s in data.bottom_students], bottom_students_widths)
        
        theory_threshold = data.processing_summary.pass_threshold_theory
        practical_threshold = data.processing_summary.pass_threshold_practical
        
        failed_students_data = []
        if data.all_students:
            for student_model in data.all_students:
                student = student_model.model_dump(exclude_none=True)
                theory_marks = student.get('Theory') 
                practical_marks = student.get('Practical')
                
                # Handle cases where thresholds or marks might be None
                is_theory_eligible_val = (theory_marks is not None and theory_threshold is not None and theory_marks >= theory_threshold)
                is_practical_eligible_val = (practical_marks is not None and practical_threshold is not None and practical_marks >= practical_threshold)
                
                if not (is_theory_eligible_val and is_practical_eligible_val):
                    failed_students_data.append(student)

        failed_students_headers = {
            "USN": "USN", 
            "Theory": "Theory", "Theory_Eligible": "Theory Eligible",
            "Practical": "Practical", "Practical_Eligible": "Practical Eligible",
            "Total": "Total", "Overall_Eligible": "Overall Eligible"
        }
        failed_students_widths = [18, 10, 14, 10, 14, 10, 24] # Adjusted for more columns
        pdf.student_table("Failed Students List", failed_students_headers, failed_students_data, failed_students_widths)

        # --- THIS IS THE CORRECTED LINE ---
        pdf_output = pdf.output(dest='S') 
        # ---------------------------------
        
        logger.info("PDF generated successfully.")
        return StreamingResponse(
            io.BytesIO(pdf_output), 
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment;filename=marks_summary_report.pdf"}
        )
    except Exception as e:
        logger.error(f"Error generating PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not generate PDF report: {str(e)}")

# If running directly (for local testing, though uvicorn is preferred for FastAPI)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)