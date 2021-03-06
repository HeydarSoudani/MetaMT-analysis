from os import path

loc = {
    "train": {
        "qa_en": ["squad/train-v1.1.json", False],
        "qa_fa": ["persianqa/pqa_train.json", False],
        "qa_de": ["squad/de_squad-translate-train-train-v1.1.json", False],
        "qa_hi": ["squad/hi_squad-translate-train-train-v1.1.json", False],
        "qa_es": ["squad/es_squad-translate-train-train-v1.1.json", False],
        "sc_en": ["xnli/train-en.tsv", "csv"],
        "sc_fa": ["farstail/Train-word.csv", "csv"],
        "sc_de": ["xnli/xnli.translate.train.clean.en-de.tsv", "csv"],
        "sc_es": ["xnli/xnli.translate.train.clean.en-es.tsv", "csv"],
        "sc_fr": ["xnli/xnli.translate.train.clean.en-fr.tsv", "csv"],
        "tc_en": ["panx/train-en.tsv"],
        "tc_hi": ["panx/train-hi.tsv"],
        "tc_de": ["panx/train-de.tsv"],
        "tc_es": ["panx/train-es.tsv"],
        "tc_fr": ["panx/train-fr.tsv"],
        "tc_zh": ["panx/train-zh.tsv"],
        "po_en": ["udpos/train-en.tsv"],
        "po_hi": ["udpos/train-hi.tsv"],
        "po_de": ["udpos/train-de.tsv"],
        "po_es": ["udpos/train-es.tsv"],
        "po_zh": ["udpos/train-zh.tsv"],
        "pa_en": ["pawsx/train-en.tsv"],
        "pa_de": ["pawsx/train-de.tsv"],
        "pa_es": ["pawsx/train-es.tsv"],
        "pa_fr": ["pawsx/train-fr.tsv"],
        "pa_zh": ["pawsx/train-zh.tsv"],
        "sc_en_0": ["dreca/train-0.csv", "csv"],
        "sc_en_1": ["dreca/train-1.csv", "csv"],
        "sc_en_2": ["dreca/train-2.csv", "csv"],
        "sc_en_3": ["dreca/train-3.csv", "csv"],
        "sc_en_4": ["dreca/train-4.csv", "csv"],
        "sc_en_5": ["dreca/train-5.csv", "csv"],
        "sc_en_6": ["dreca/train-6.csv", "csv"],
        "sc_en_7": ["dreca/train-7.csv", "csv"],
    },
    "dev": {
        "qa_en": ["mlqa/MLQA_V1/dev/dev-context-en-question-en.json", True],
        "qa_fa": ["persianqa/pqa_dev.json", False],
        "qa_de": ["mlqa/MLQA_V1/dev/dev-context-de-question-de.json", True],
        "qa_hi": ["mlqa/MLQA_V1/dev/dev-context-hi-question-hi.json", True],
        "qa_es": ["mlqa/MLQA_V1/dev/dev-context-es-question-es.json", True],
        "sc_en": ["xnli/dev-en.tsv", "csv"],
        "sc_fa": ["farstail/Val-word.csv", "csv"],
        "sc_es": ["xnli/dev-es.tsv", "csv"],
        "sc_de": ["xnli/dev-de.tsv", "csv"],
        "sc_fr": ["xnli/dev-fr.tsv", "csv"],
        "tc_en": ["panx/dev-en.tsv"],
        "tc_hi": ["panx/dev-hi.tsv"],
        "tc_de": ["panx/dev-de.tsv"],
        "tc_es": ["panx/dev-es.tsv"],
        "tc_fr": ["panx/dev-fr.tsv"],
        "tc_zh": ["panx/dev-zh.tsv"],
        "po_en": ["udpos/dev-en.tsv"],
        "po_hi": ["udpos/dev-hi.tsv"],
        "po_de": ["udpos/dev-de.tsv"],
        "po_es": ["udpos/dev-es.tsv"],
        "po_zh": ["udpos/dev-zh.tsv"],
        "pa_en": ["pawsx/dev-en.tsv"],
        "pa_de": ["pawsx/dev-de.tsv"],
        "pa_es": ["pawsx/dev-es.tsv"],
        "pa_fr": ["pawsx/dev-fr.tsv"],
        "pa_zh": ["pawsx/dev-zh.tsv"],
        "sc_en_0": ["xnli/dev-en.tsv", "csv"],
        "sc_en_1": ["xnli/dev-en.tsv", "csv"],
        "sc_en_2": ["xnli/dev-en.tsv", "csv"],
        "sc_en_3": ["xnli/dev-en.tsv", "csv"],
        "sc_en_4": ["xnli/dev-en.tsv", "csv"],
        "sc_en_5": ["xnli/dev-en.tsv", "csv"],
        "sc_en_6": ["xnli/dev-en.tsv", "csv"],
        "sc_en_7": ["xnli/dev-en.tsv", "csv"],
    },
    "test": {
        "qa_en": ["mlqa/MLQA_V1/test/test-context-en-question-en.json", True],
        "qa_fa": ["persianqa/pqa_test.json", False],
        "qa_de": ["mlqa/MLQA_V1/test/test-context-de-question-de.json", True],
        "qa_hi": ["mlqa/MLQA_V1/test/test-context-hi-question-hi.json", True],
        "qa_es": ["mlqa/MLQA_V1/test/test-context-es-question-es.json", True],
        "sc_en": ["xnli/test-en.tsv", "csv"],
        "sc_fa": ["farstail/Test-word.csv", "csv"],
        "sc_es": ["xnli/test-es.tsv", "csv"],
        "sc_de": ["xnli/test-de.tsv", "csv"],
        "sc_fr": ["xnli/test-fr.tsv", "csv"],
        "tc_en": ["panx/test-en.tsv"],
        "tc_hi": ["panx/test-hi.tsv"],
        "tc_de": ["panx/test-de.tsv"],
        "tc_es": ["panx/test-es.tsv"],
        "tc_fr": ["panx/test-fr.tsv"],
        "tc_zh": ["panx/test-zh.tsv"],
        "po_en": ["udpos/test-en.tsv"],
        "po_hi": ["udpos/test-hi.tsv"],
        "po_de": ["udpos/test-de.tsv"],
        "po_es": ["udpos/test-es.tsv"],
        "po_zh": ["udpos/test-zh.tsv"],
        "pa_en": ["pawsx/test-en.tsv"],
        "pa_de": ["pawsx/test-de.tsv"],
        "pa_es": ["pawsx/test-es.tsv"],
        "pa_fr": ["pawsx/test-fr.tsv"],
        "pa_zh": ["pawsx/test-zh.tsv"],
        "qa_ar": ["mlqa/MLQA_V1/test/test-context-ar-question-ar.json", True],
        "qa_vi": ["mlqa/MLQA_V1/test/test-context-vi-question-vi.json", True],
        "sc_ar": ["xnli/test-ar.tsv", "csv"],
        "sc_bg": ["xnli/test-bg.tsv", "csv"],
        "sc_el": ["xnli/test-el.tsv", "csv"],
        "sc_ru": ["xnli/test-ru.tsv", "csv"],
        "sc_sw": ["xnli/test-sw.tsv", "csv"],
        "sc_th": ["xnli/test-th.tsv", "csv"],
        "sc_tr": ["xnli/test-tr.tsv", "csv"],
        "sc_ur": ["xnli/test-ur.tsv", "csv"],
        "sc_vi": ["xnli/test-vi.tsv", "csv"],
        "pa_ja": ["pawsx/test-ja.tsv"],
        "pa_ko": ["pawsx/test-ko.tsv"],
        "po_mr": ["udpos/test-mr.tsv"],
        "po_ur": ["udpos/test-ur.tsv"],
        "po_ta": ["udpos/test-ta.tsv"],
        "po_te": ["udpos/test-te.tsv"],
        "po_ja": ["udpos/test-ja.tsv"],
        "po_et": ["udpos/test-et.tsv"],
        "po_fi": ["udpos/test-fi.tsv"],
        "tc_bn": ["panx/test-bn.tsv"],
        "tc_mr": ["panx/test-mr.tsv"],
        "tc_ur": ["panx/test-ur.tsv"],
        "tc_ta": ["panx/test-ta.tsv"],
        "tc_te": ["panx/test-te.tsv"],
        "tc_ja": ["panx/test-ja.tsv"],
        "tc_et": ["panx/test-et.tsv"],
        "tc_fi": ["panx/test-fi.tsv"],
    },
}


def get_loc(type, task, base_dir="data/"):
    this_loc = loc[type][task]
    return [path.join(base_dir, this_loc[0]), this_loc[1]]
