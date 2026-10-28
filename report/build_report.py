"""Build the submission report PDF with ReportLab."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "mp2-report.pdf"

pdfmetrics.registerFont(TTFont("TimesNewRoman", r"C:\Windows\Fonts\times.ttf"))
pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", r"C:\Windows\Fonts\timesbd.ttf"))
pdfmetrics.registerFont(TTFont("TimesNewRoman-Italic", r"C:\Windows\Fonts\timesi.ttf"))
pdfmetrics.registerFont(TTFont("TimesNewRoman-BoldItalic", r"C:\Windows\Fonts\timesbi.ttf"))


class NumberedDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=self._decorate))

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.black)
        canvas.setFont("TimesNewRoman", 9)
        canvas.drawCentredString(A4[0] / 2, 12 * mm, str(doc.page))
        canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName="TimesNewRoman-Bold",
            fontSize=17,
            leading=21,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="TimesNewRoman",
            fontSize=11,
            leading=14,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1Custom",
            parent=base["Heading1"],
            fontName="TimesNewRoman-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.black,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2Custom",
            parent=base["Heading2"],
            fontName="TimesNewRoman-Bold",
            fontSize=11,
            leading=13,
            textColor=colors.black,
            spaceBefore=5,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="TimesNewRoman",
            fontSize=9.5,
            leading=12,
            alignment=TA_JUSTIFY,
            textColor=colors.black,
            spaceAfter=3,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=base["BodyText"],
            fontName="TimesNewRoman",
            fontSize=9.5,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.black,
            leftIndent=8 * mm,
            rightIndent=8 * mm,
            spaceBefore=3,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="TimesNewRoman",
            fontSize=8.5,
            leading=11,
            textColor=colors.black,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="TimesNewRoman",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceBefore=2,
            spaceAfter=4,
        ),
    }


def p(text, style):
    return Paragraph(text, style)


def section(title, paragraphs, sheet):
    items = [p(title, sheet["h1"])]
    items.extend(p(text, sheet["body"]) for text in paragraphs)
    return items


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet = styles()
    doc = NumberedDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title="Selecting Reliable LLM Reasoning with a Neuro-Symbolic Trace Verifier",
        author="DASE7506 MP2 submission",
        subject="DASE7506 Advanced Machine Learning Methods - Mini-project 2",
    )

    story = [
        Spacer(1, 5 * mm),
        p("Selecting Reliable LLM Reasoning<br/>with a Neuro-Symbolic Trace Verifier", sheet["title"]),
        p(
            "DASE7506 Advanced Machine Learning Methods - Mini-project 2",
            sheet["subtitle"],
        ),
    ]

    story += section(
        "1. Research question and hypothesis",
        [
            "The main questions are: <b>Do representations of full candidate reasoning help "
            "beyond answer agreement; should candidates be judged comparatively; and when does "
            "additional computation produce a reliable gain?</b>",
            "The hypothesis is that majority agreement is a strong signal, but candidates often "
            "share the same semantic mistake. A learned verifier using the supplied whole-trace "
            "embeddings should detect additional patterns associated with valid reasoning. A "
            "low-rank nonlinear projection should capture interactions missed by a linear model, "
            "while explicit equation checks should identify arithmetic failures.",
            "Parameters are fitted only from training inputs and references. Validation is used "
            "for model selection. Each reported version was scored only after its configuration "
            "was frozen. V32 was selected from repeated training cross-validation and validation "
            "evidence without using any test outcome for training, tuning, or model selection.",
        ],
        sheet,
    )

    story += section(
        "2. Data and metric",
        [
            "The supplied bank contains 4,755 training, 798 validation, and 467 test questions. "
            "Each question has four candidate reasoning traces and a 1,024-dimensional "
            "representation for each trace. The output is exactly one candidate index per "
            "question. The official scorer parses the selected candidate's final numeric answer "
            "and awards a point when it equals the reference answer; it does not assess every "
            "reasoning step.",
            "There are 379 validation questions on which all four parsed answers agree and 419 "
            "on which they disagree. Since every group contains at least one correct answer, all "
            "candidates in a unanimous group receive the same score. I report both complete "
            "validation accuracy and disagreement-only accuracy. The latter isolates questions "
            "where selection changes the outcome and is more comparable with the test bank, "
            "where every group contains at least two distinct valid parsed answers.",
        ],
        sheet,
    )

    story += [p("3. Method", sheet["h1"]), p("3.1 Baselines", sheet["h2"])]
    story.append(
        p(
            "The first-position baseline always selects candidate 0. The answer-only baseline "
            "parses answers with the supplied numeric parser, groups equal fractions and decimals, "
            "and chooses the largest valid group. Ties go to the earliest candidate. This majority "
            "method uses no reasoning representation.",
            sheet["body"],
        )
    )
    story += [p("3.2 Surface-feature verifier", sheet["h2"])]
    story.append(
        p(
            "For each candidate I extract 23 small, auditable features: answer validity and group "
            "size, candidate position, text/word/line and number counts, coverage of numbers "
            "mentioned in the question, answer magnitude and integrality, final-answer markers, "
            "step/equation counts, and conclusion markers. This ablation tests how much can be "
            "gained from shallow formatting and lexical cues.",
            sheet["body"],
        )
    )
    story += [p("3.3 Relative embedding verifier", sheet["h2"])]
    story.append(
        p(
            "For candidate j of question i, I subtract the mean of that question's four supplied "
            "embeddings and L2-normalize the result:",
            sheet["body"],
        )
    )
    story.append(p("r_ij = (e_ij - mean_k(e_ik)) / ||e_ij - mean_k(e_ik)||_2", sheet["equation"]))
    story.append(
        p(
            "This question-relative representation focuses the linear verifier on how a candidate "
            "differs from its alternatives. Each dimension is standardized with training statistics. "
            "A ridge model predicts whether a candidate's parsed answer equals the training reference:",
            sheet["body"],
        )
    )
    story.append(p("w_hat = argmin_w ||Xw - y||_2^2 + alpha ||w||_2^2", sheet["equation"]))
    story.append(
        p(
            "The embedding score is combined with an independently trained surface score:",
            sheet["body"],
        )
    )
    story.append(p("b_j = scale(s_j^(e) + 0.25 s_j^(u))", sheet["equation"]))
    story += [p("3.4 Nonlinear and arithmetic verifiers", sheet["h2"])]
    story.append(
        p(
            "A fixed 1,024 by 128 Gaussian projection generated with seed 1 supplies low-rank "
            "nonlinear interactions in the relative embedding:",
            sheet["body"],
        )
    )
    story.append(p("z_ij = tanh(0.5 r_ij R)", sheet["equation"]))
    story.append(
        p(
            "A separate ridge model scores these features. The arithmetic verifier safely parses "
            "numeric equation chains written in each trace. A tolerant companion accepts common "
            "LaTeX and unit suffixes. A 31-feature block also describes answer rank, distance from "
            "the group median, sign, integrality, scale, and simple arithmetic reachability. Ridge "
            "and logistic models score these features. Candidate text is never executed. The "
            "scaled V6 component scores are fused as:",
            sheet["body"],
        )
    )
    story.append(p("s_j = b_j + 0.475 s_j^(n) + 0.1 s_j^(a1) + 0.25 s_j^(a2) + 0.5 s_j^(r) + 0.9 s_j^(l)", sheet["equation"]))
    story.append(
        p(
            "At inference, candidates with unparseable answers are masked. Candidates with the "
            "same parsed numeric answer form a group. Its score combines the strongest candidate "
            "evidence with a scale-stable corroboration term:",
            sheet["body"],
        )
    )
    story.append(Spacer(1, 1.5 * mm))
    story.append(p("g(A) = max_(j: a_j=A) s_j + 0.275 log |{j: a_j=A}|", sheet["equation"]))
    story.append(
        p(
            "The best group is selected, followed by its highest-scoring candidate. The procedure accepts arbitrary bank sizes and "
            "never opens reference files.",
            sheet["body"],
        )
    )
    story.append(
        p(
            "V9 fits a 128-dimensional PCA basis to normalized question-relative trace embeddings; "
            "the retained axes explain 87.96% of training variance. It concatenates V6 group "
            "evidence with mean and maximum PCA trace features. A class-balanced RBF SVM with "
            "C=1 is trained on out-of-fold group features and contributes a 1.5-weight score. "
            "The PCA basis and 2,547 support vectors are evaluated with NumPy at inference.",
            sheet["body"],
        )
    )
    story += [p("3.5 Bounded symbolic program search", sheet["h2"])]
    story.append(
        p(
            "V32 extracts up to eight quantities from the question and constructs arithmetic "
            "programs with addition, subtraction, multiplication, and division. Programs use at "
            "most five input occurrences, while a target-aware beam retains at most 60 values "
            "per number subset. This bounds runtime without using question IDs or references.",
            sheet["body"],
        )
    )
    story.append(
        p(
            "Nineteen candidate features summarize exact reachability, direct occurrence, program "
            "size, input coverage, operation types, reachable-program counts, closest-value "
            "distance, and integrality. Mean and maximum features are pooled within equal-answer "
            "groups and combined with compact semantic and executed-trace evidence by a 12-unit "
            "tanh listwise ranker:",
            sheet["body"],
        )
    )
    story.append(p("s_j^(V32) = s_j^(V9) + 1.5 s_j^(program)", sheet["equation"]))
    story.append(
        p(
            "The search does not reproduce a natural-language rationale. It tests the narrower "
            "and more robust condition that an answer can be constructed by a short arithmetic "
            "program over quantities explicitly present in the problem.",
            sheet["body"],
        )
    )
    story += [p("3.6 Selection protocol", sheet["h2"])]
    story.append(
        p(
            "I compare surface features, embeddings, their concatenation, candidate-level "
            "selection, answer-group aggregation, direct listwise training, late fusion, random "
            "nonlinear features, arithmetic features, numeric structure, alternative targets, "
            "ensembles, nearest-neighbour and lexical models, stacking, wider kernels, pairwise "
            "ranking, executed-trace provenance, consensus selectors, reference-plan prediction, "
            "and train-only easy-positive augmentation. New components are admitted only when "
            "validation and repeated training-only comparisons support them. V32 produced the "
            "strongest full validation result and improved V9 on informative questions under all "
            "three repeated partitions. Its configuration was frozen before public-test scoring.",
            sheet["body"],
        )
    )

    story.append(p("4. Results", sheet["h1"]))
    table_data = [
        ["Method", "Val. correct", "Val. acc.", "Disagree correct", "Disagree acc."],
        ["First candidate", "624/798", "78.20%", "245/419", "58.47%"],
        ["Answer majority", "688/798", "86.22%", "309/419", "73.75%"],
        ["Surface features", "706/798", "88.47%", "327/419", "78.04%"],
        ["Candidate embedding", "727/798", "91.10%", "348/419", "83.05%"],
        ["Answer-group embedding", "732/798", "91.73%", "353/419", "84.25%"],
        ["Surface + embedding", "726/798", "90.98%", "347/419", "82.82%"],
        ["Late-fused answer group (V3)", "736/798", "92.23%", "357/419", "85.20%"],
        ["Nonlinear embedding fusion", "740/798", "92.73%", "361/419", "86.16%"],
        ["Nonlinear + arithmetic (V4)", "742/798", "92.98%", "363/419", "86.63%"],
        ["Structured ridge (V5)", "746/798", "93.48%", "367/419", "87.59%"],
        ["Ridge + logistic numeric (V6)", "746/798", "93.48%", "367/419", "87.59%"],
        ["Random-projection RBF (V8)", "747/798", "93.61%", "368/419", "87.83%"],
        ["PCA semantic RBF (V9)", "748/798", "93.73%", "369/419", "88.07%"],
        ["Program search + listwise (V32)", "752/798", "94.24%", "373/419", "89.02%"],
    ]
    story.append(p("Table 1. Validation accuracy and disagreement-only accuracy.", sheet["caption"]))
    result_table = Table(table_data, colWidths=[49 * mm, 25 * mm, 23 * mm, 31 * mm, 27 * mm], repeatRows=1)
    result_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "TimesNewRoman-Bold"),
                ("FONTNAME", (0, 1), (-1, -2), "TimesNewRoman"),
                ("FONTNAME", (0, -1), (-1, -1), "TimesNewRoman-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("ALIGN", (1, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story += [result_table, Spacer(1, 3 * mm)]
    story += [
        p(
            "V32 improves over majority by 64 validation questions, or 8.02 percentage points; "
            "on disagreement questions the gain is 15.27 points. It improves V9 by four "
            "validation questions, all from the disagreement subset. The surface-only model "
            "trails the embedding verifier by 2.63 points, so formatting cues do not account "
            "for the main gain.",
            sheet["body"],
        ),
        p(
            "The V32 program-search ranker improves V9 on informative train/CV questions by "
            "+8, +7, and +5 under the default, salt-A, and salt-B partitions; corresponding "
            "informative validation gains are +4, +6, and +5. After freezing the configuration, "
            "the official public-test result is <b>392/467 = 83.940043%</b>. Test examples and "
            "per-example outcomes are not used for training or parameter selection.",
            sheet["body"],
        ),
    ]

    story.append(p("4.1 Controlled ablation study", sheet["h2"]))
    story.append(
        p(
            "Table 2 reports matched comparisons that isolate key design choices. Positive "
            "deltas mean that restoring the named component improves over its ablated method; "
            "the rows are controlled pairs rather than an undifferentiated version history. "
            "Every pair uses the same frozen validation split and scorer. Because all 379 "
            "unanimous-answer groups are correct under any selection, full-set count changes "
            "equal disagreement-set count changes; the latter is the more sensitive metric.",
            sheet["body"],
        )
    )
    ablation_data = [
        ["Component restored", "Ablated", "With component", "Val. delta", "Disagree delta"],
        ["Full-trace representation", "Surface: 706", "Embedding: 727", "+21 (+2.63 pp)", "+21 (+5.01 pp)"],
        ["Answer-group comparison", "Candidate: 727", "Grouped: 732", "+5 (+0.63 pp)", "+5 (+1.19 pp)"],
        ["Arithmetic checking", "Nonlinear: 740", "V4: 742", "+2 (+0.25 pp)", "+2 (+0.48 pp)"],
        ["Logistic calibration", "V5: 746", "V6: 746", "0 (0.00 pp)", "0 (0.00 pp)"],
        ["Bounded symbolic module", "V9: 748", "V32: 752", "+4 (+0.50 pp)", "+4 (+0.95 pp)"],
    ]
    story.append(p("Table 2. Controlled component ablations on validation data.", sheet["caption"]))
    ablation_table = Table(
        ablation_data,
        colWidths=[42 * mm, 28 * mm, 31 * mm, 27 * mm, 27 * mm],
        repeatRows=1,
    )
    ablation_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "TimesNewRoman-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "TimesNewRoman"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story += [ablation_table, Spacer(1, 2.5 * mm)]
    story += [
        p(
            "<b>Reasoning representation versus surface cues.</b> Replacing the 23 shallow "
            "features with the supplied full-trace representation adds 21 correct choices. On "
            "disagreement questions, errors fall from 92 to 71, a 22.8% relative reduction. "
            "Length, formatting, equation counts, and conclusion markers therefore do not "
            "explain most of the gain. This does not prove that every embedding dimension encodes "
            "valid reasoning: latent stylistic cues may remain. It establishes the narrower claim "
            "that the whole-trace representation contains information beyond the audited surface block.",
            sheet["body"],
        ),
        p(
            "<b>Comparative judgement.</b> Answer grouping raises the result from 727 to 732 "
            "correct. All five recovered decisions are disagreements, reducing their errors from "
            "71 to 66 (7.0%). This supports the hypothesis that alternatives provide useful "
            "context. Grouping cannot help unanimous groups and may reinforce a shared mistake, "
            "so the method treats agreement as corroboration rather than a correctness rule.",
            sheet["body"],
        ),
        p(
            "<b>Arithmetic evidence and model capacity.</b> The strict arithmetic checker adds "
            "two correct decisions and reduces disagreement errors from 58 to 56 (3.4%). Its "
            "coverage is limited to equations explicitly present in a trace and cannot expose a "
            "misread condition followed by consistent arithmetic. Logistic numeric calibration "
            "in V6 changes no top-1 decision. Its score may still inform later fusion, but the "
            "ablation gives no evidence for an independent accuracy gain and shows that added "
            "nonlinear capacity need not help.",
            sheet["body"],
        ),
        p(
            "<b>Bounded symbolic module.</b> V9 is the ablation that removes program-search "
            "features and their 12-unit listwise scorer. V32 changes 11 selected answers: seven "
            "repair an error, three introduce an error, and one leaves correctness unchanged, "
            "for a net gain of four. Disagreement errors fall from 50 to 46, an 8.0% relative "
            "reduction. The regressions show that reachability is evidence, not proof.",
            sheet["body"],
        ),
        p(
            "The single-split gain is small, so repeated evidence matters. Under the default, "
            "salt-A, and salt-B partitions, V32 improves informative train/CV questions over V9 "
            "by +8, +7, and +5; corresponding informative-validation gains are +4, +6, and +5. "
            "The consistent direction is stronger than the +4 headline alone, although repeated "
            "use of the same validation bank still leaves some model-selection bias.",
            sheet["body"],
        ),
        p(
            "<b>Accuracy-cost trade-off and scope.</b> Fresh program-feature search takes 257.07 "
            "seconds, 84.0% of the 306.11-second clean reproduction process. The frozen V32 asset "
            "is 4.78 MiB and complete local prediction plus scoring takes 18.23 seconds. The "
            "accuracy return is modest for most of the reproduction-time computation, but the "
            "module is retained because gains repeat and deployment stays within the 16 MiB and "
            "30-second limits. This ablation removes search and learned fusion together; it does "
            "not attribute their contributions separately.",
            sheet["body"],
        ),
    ]

    resource_section = section(
        "5. Resource use and reproducibility",
        [
            "Training uses fixed seeds 1 (random projection and PCA) and 8200 (listwise "
            "ranker). On an AMD Ryzen 9 8945HX, a clean base train, V9 export, fresh V32 "
            "search/cache build, and V32 export take 17.84, 13.20, 257.07, and 19.00 seconds "
            "respectively (306.11 seconds total). The regenerated model produces identical "
            "validation and test choices. With OMP, OpenBLAS, and MKL thread counts fixed to "
            "four, fresh V32 prediction followed by independent scoring takes 18.23 seconds. "
            "The reference Xeon remains authoritative.",
            "The only inference asset is model.npz (5,013,724 bytes, 4.78 MiB). Its SHA-256 is<br/>"
            "<font name='Courier' size='7'>b384830a79de9d98e204d4273987b0ea7afaf0e84c008bdb0758a06d610af870</font>. "
            "It stores the PCA basis, compact verifiers, and their training statistics. The method uses only NumPy at "
            "inference and performs no network access, generation, retraining, ID lookup, or "
            "assumption about bank size.",
        ],
        sheet,
    )
    story += [KeepTogether(resource_section)]

    story += section(
        "6. Limitations and error analysis",
        [
            "The supplied embedding summarizes the whole trace, so the experiment shows that the "
            "representation is useful but cannot identify which individual reasoning step caused "
            "a score. The surface ablation reduces one alternative explanation, yet unmeasured "
            "stylistic cues may remain embedded. A stronger process-level study would need "
            "step-level representations or a permitted way to recompute embeddings after "
            "controlled trace masking; neither is supplied here.",
            "The arithmetic checker validates equations that candidates explicitly write, but a "
            "trace can remain self-consistent while misreading the problem or omitting a condition. "
            "The bounded search is intentionally incomplete: it uses the first eight quantities, "
            "at most five inputs, four elementary operations, rounded states, and a finite beam. "
            "It does not directly model units, implicit constants, language-only rates, or longer "
            "programs. Test accuracy remains below validation disagreement accuracy, indicating "
            "distribution shift. Future work should add unit- and relation-aware symbolic states.",
        ],
        sheet,
    )

    story += section(
        "7. Conclusion",
        [
            "A 4.78 MiB verifier combining relative full-trace embeddings, PCA semantic routing, "
            "arithmetic checks, numeric structure, parsed-answer grouping, and bounded symbolic "
            "search substantially outperforms fixed-position and majority baselines. The final "
            "V32 selector reaches 94.24% validation and 83.94% frozen public-test accuracy within "
            "the CPU and asset limits.",
        ],
        sheet,
    )

    story += [
        p(
            "<b>AI assistance and reuse disclosure.</b> OpenAI Codex provided substantive "
            "assistance in interpreting the assignment, "
            "implementing and testing the method, running the documented experiments, and "
            "drafting this report. The submitter reviewed the resulting code and remains "
            "responsible for understanding and explaining it. Frozen inference uses NumPy, while "
            "training additionally uses scikit-learn. No external dataset, pretrained weight, "
            "third-party solution, or additional model asset is included.",
            sheet["small"],
        ),
    ]

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
