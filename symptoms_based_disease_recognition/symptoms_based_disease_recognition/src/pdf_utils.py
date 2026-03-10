from fpdf import FPDF


class _ReportPDF(FPDF):
    def header(self):
        # Keep default header empty; we draw the full header in content.
        return


def generate_prediction_report(
    report_id: str,
    date_time: str,
    patient_name: str,
    symptoms: list,
    predicted_disease: str,
    recommended_tests: list,
):
    """
    Generates a PDF report visually similar to the sample screenshot:
    - Lab header (left title, right contact)
    - 2x2 info boxes (report id, date/time, patient, type)
    - Symptoms list
    - Diagnosis highlight
    - Recommended tests list
    - Footer disclaimer + lab stamp (bottom-right)
    """

    pdf = _ReportPDF()
    # Keep everything on a single page for typical reports
    pdf.set_auto_page_break(auto=False, margin=18)
    pdf.add_page()

    # Page constants
    left = 14
    right = 14
    page_w = pdf.w
    usable_w = page_w - left - right

    # Colors
    accent_r, accent_g, accent_b = 240, 47, 52  # red
    border_r, border_g, border_b = 210, 218, 229
    muted_r, muted_g, muted_b = 110, 120, 135
    stamp_r, stamp_g, stamp_b = 34, 139, 34  # green

    # -------------------------
    # Header
    # -------------------------
    pdf.set_xy(left, 12)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 7, "DRLOGY PATHOLOGY LAB", ln=1)

    pdf.set_x(left)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(muted_r, muted_g, muted_b)
    pdf.cell(0, 5, "Accurate | Caring | Instant", ln=1)

    # Right header contact
    pdf.set_xy(left, 12)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(muted_r, muted_g, muted_b)
    pdf.set_xy(left + usable_w * 0.60, 12)
    pdf.multi_cell(
        usable_w * 0.40,
        4,
        "Phone: 0123456789 / 0912345678\nEmail: drlogypathlab@gmail.com\nwww.drlogy.com",
        align="R",
    )

    # Header separator line
    pdf.set_draw_color(border_r, border_g, border_b)
    pdf.set_line_width(0.6)
    pdf.line(left, 32, page_w - right, 32)

    y = 38

    # -------------------------
    # Info boxes (2x2)
    # -------------------------
    box_h = 16
    gap = 6
    box_w = (usable_w - gap) / 2

    def info_box(x, y, label, value):
        pdf.set_draw_color(border_r, border_g, border_b)
        pdf.set_fill_color(255, 255, 255)
        pdf.rect(x, y, box_w, box_h, style="D")
        pdf.set_xy(x + 4, y + 3)
        pdf.set_font("Arial", "", 9)
        pdf.set_text_color(muted_r, muted_g, muted_b)
        pdf.cell(box_w - 8, 4, label, ln=1)
        pdf.set_x(x + 4)
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(box_w - 8, 7, value, ln=1)

    info_box(left, y, "Report / Receipt ID", str(report_id)[:12].upper())
    info_box(left + box_w + gap, y, "Date & Time", date_time)
    y += box_h + 8
    info_box(left, y, "Patient", patient_name)
    info_box(left + box_w + gap, y, "Type", "Disease Prediction Report")
    y += box_h + 14

    # -------------------------
    # Symptoms
    # -------------------------
    pdf.set_xy(left, y)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, "Symptoms", ln=1)
    y = pdf.get_y() + 2

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    if symptoms:
        # Limit to first 10 symptoms to keep layout on one page
        for s in symptoms[:10]:
            pdf.set_x(left + 2)
            pdf.cell(4, 6, chr(149), ln=0)  # bullet
            pdf.multi_cell(0, 6, str(s))
    else:
        pdf.set_x(left)
        pdf.set_text_color(muted_r, muted_g, muted_b)
        pdf.cell(0, 6, "No symptoms recorded.", ln=1)

    y = pdf.get_y() + 6

    # -------------------------
    # Diagnosis
    # -------------------------
    pdf.set_xy(left, y)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, "Diagnosis", ln=1)
    y = pdf.get_y() + 2

    # Diagnosis highlight box
    pdf.set_draw_color(border_r, border_g, border_b)
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(left, y, usable_w, 14, style="D")
    pdf.set_xy(left + 6, y + 4.5)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(accent_r, accent_g, accent_b)
    pdf.cell(0, 6, predicted_disease, ln=1)
    y += 20

    # -------------------------
    # Recommended tests
    # -------------------------
    pdf.set_xy(left, y)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, "Recommended Tests", ln=1)
    y = pdf.get_y() + 2

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    if recommended_tests:
        # Limit to first 6 tests to keep layout compact
        for t in recommended_tests[:6]:
            pdf.set_x(left + 2)
            pdf.cell(4, 6, chr(149), ln=0)
            pdf.multi_cell(0, 6, str(t))
    else:
        pdf.set_text_color(muted_r, muted_g, muted_b)
        pdf.set_x(left)
        pdf.cell(0, 6, "No specific tests recommended.", ln=1)

    # -------------------------
    # Footer disclaimer + stamp
    # -------------------------
    pdf.set_text_color(muted_r, muted_g, muted_b)
    pdf.set_font("Arial", "", 9)

    # Position near bottom
    footer_y = pdf.h - 28
    pdf.set_xy(left, footer_y)
    pdf.multi_cell(
        usable_w * 0.60,
        4,
        "This is a computer generated report for reference only.\nPlease consult a doctor for medical advice.",
    )

    # Draw a simple green stamp at bottom-right
    stamp_w = 52
    stamp_h = 26
    stamp_x = page_w - right - stamp_w
    stamp_y = pdf.h - 34
    pdf.set_draw_color(stamp_r, stamp_g, stamp_b)
    pdf.set_line_width(1.2)
    pdf.rect(stamp_x, stamp_y, stamp_w, stamp_h, style="D")
    pdf.set_text_color(stamp_r, stamp_g, stamp_b)
    pdf.set_font("Arial", "B", 11)
    pdf.set_xy(stamp_x, stamp_y + 5)
    pdf.cell(stamp_w, 6, "LAB STAMP", align="C", ln=1)
    pdf.set_font("Arial", "B", 9)
    pdf.set_x(stamp_x)
    pdf.cell(stamp_w, 6, "DRLOGY", align="C", ln=1)
    pdf.set_font("Arial", "", 8)
    pdf.set_x(stamp_x)
    pdf.cell(stamp_w, 5, "PATHOLOGY LAB", align="C", ln=1)

    return pdf
