#!/usr/bin/env python3

import unittest

from ar_XA import local_nav_diacritizer as nav


class LocalNavDiacritizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_tashkeel = nav.tashkeel_run
        nav.tashkeel_run = lambda text: text

    def tearDown(self) -> None:
        nav.tashkeel_run = self.original_tashkeel

    def test_word_ngram_does_not_skip_punctuation(self) -> None:
        key = nav.make_key(["الأول", "الثاني"])
        value = nav.make_key(["اَلْأَوَّلُ", "اَلثَّانِيِ"])
        engine = nav.LocalNavDiacritizer(
            {"line_rules": {}, "ngram_rules": {"2": {key: value}}},
            {},
        )

        self.assertEqual(
            engine.diacritize_line("الأول الثاني."),
            "اَلْأَوَّلُ اَلثَّانِيْ.",
        )
        self.assertEqual(
            engine.diacritize_line("الأول، الثاني."),
            "الأول، الثاني.",
        )

    def test_ngram_can_match_across_punctuation_when_rule_includes_it(self) -> None:
        key = nav.make_key(["الأول", "،", "الثاني"])
        value = nav.make_key(["اَلْأَوَّلُ", "،", "اَلثَّانِيِ"])
        engine = nav.LocalNavDiacritizer(
            {"line_rules": {}, "ngram_rules": {"3": {key: value}}},
            {},
        )

        self.assertEqual(
            engine.diacritize_line("الأول، الثاني."),
            "اَلْأَوَّلُ، اَلثَّانِيْ.",
        )

    def test_normalize_converts_exclamation_marks_to_periods(self) -> None:
        self.assertEqual(
            nav.normalize_text("توقف!ثم تابع！"),
            "توقف. ثم تابع.",
        )

    def test_fixed_phrase_uses_contextual_fahd_ending(self) -> None:
        engine = nav.LocalNavDiacritizer(
            {"line_rules": {}, "ngram_rules": {}},
            {},
        )

        self.assertEqual(
            engine.diacritize_line("إلى شارع الملك فهد نحو الرياض."),
            "إِلَى شَارِعِ الْمَلِكِ فَهْدٍ نحو الرياض.",
        )
        self.assertEqual(
            engine.diacritize_line("إلى شارع الملك فهد."),
            "إِلَى شَارِعِ الْمَلِكِ فَهْدْ.",
        )

    def test_latin_mapping_can_use_line_rule(self) -> None:
        plain = "مخرج إيه."
        expected = "مَخْرَجُ إِيه."
        engine = nav.LocalNavDiacritizer(
            {"line_rules": {plain: expected}, "ngram_rules": {}},
            {"A": "إِيه"},
        )

        self.assertEqual(engine.diacritize_line("مخرج A."), expected)

    def test_only_words_created_from_latin_mapping_are_protected(self) -> None:
        key = nav.make_key(["بي"])
        engine = nav.LocalNavDiacritizer(
            {
                "line_rules": {},
                "ngram_rules": {"1": {key: nav.make_key(["بِيَ"])}},
            },
            {"B": "بِي"},
        )

        self.assertEqual(engine.diacritize_line("B،"), "بِي،")
        self.assertEqual(engine.diacritize_line("بِي،"), "بِيَ،")

    def test_multiword_latin_mapping_protects_every_created_word(self) -> None:
        engine = nav.LocalNavDiacritizer(
            {"line_rules": {}, "ngram_rules": {}},
            {"W": "دَبْلْ يُو"},
        )

        self.assertEqual(engine.diacritize_line("W،"), "دَبْلْ يُو،")

    def test_latin_mapping_only_expands_standalone_letters(self) -> None:
        text, protected = nav.replace_latin_letters(
            "A AB BA B",
            {"A": "إِيه", "B": "بِي"},
        )

        self.assertEqual(text, "إِيه AB BA بِي")
        self.assertEqual(protected, {0, 1})

    def test_waqf_preserves_existing_final_vowel(self) -> None:
        engine = nav.LocalNavDiacritizer(
            {"line_rules": {}, "ngram_rules": {}},
            {},
        )

        self.assertEqual(engine.diacritize_line("الطريقِ."), "الطريقِ.")

    def test_punctuation_ngrams_distinguish_period_and_comma(self) -> None:
        rules = {
            "line_rules": {},
            "ngram_rules": {
                "2": {
                    nav.make_key(["فهد", "."]): nav.make_key(["فَهْدْ", "."]),
                    nav.make_key(["فهد", "،"]): nav.make_key(["فَهْدٍ", "،"]),
                },
            },
        }
        engine = nav.LocalNavDiacritizer(rules, {})

        self.assertEqual(engine.diacritize_line("فهد."), "فَهْدْ.")
        self.assertEqual(engine.diacritize_line("فهد،"), "فَهْدٍ،")

    def test_grouped_split_has_no_plain_text_overlap(self) -> None:
        texts = [
            "اَلْأَوَّلُ.",
            "اَلْأَوَّلَ.",
            "اَلثَّانِي.",
            "اَلثَّالِثُ.",
        ]
        train, test = nav.split_grouped_indexes(texts, train_size=2, seed=7)
        train_plain = {nav.strip_diacritics(texts[idx]) for idx in train}
        test_plain = {nav.strip_diacritics(texts[idx]) for idx in test}

        self.assertFalse(train_plain & test_plain)

    def test_fully_covered_line_skips_tashkeel(self) -> None:
        nav.tashkeel_run = lambda text: self.fail("unexpected Tashkeel call")
        key = nav.make_key(["انعطف"])
        engine = nav.LocalNavDiacritizer(
            {
                "line_rules": {},
                "ngram_rules": {"1": {key: nav.make_key(["اِنْعَطِفْ"])}},
            },
            {},
        )

        self.assertEqual(engine.diacritize_line("انعطف."), "اِنْعَطِفْ.")

    def test_unigram_uses_only_unigram_ngram_rules(self) -> None:
        """Single-word rows have no cross-word context; n>=2 rules must not apply."""
        key1 = nav.make_key(["كلمة"])
        key2 = nav.make_key(["كلمة", "ثانية"])
        engine = nav.LocalNavDiacritizer(
            {
                "line_rules": {},
                "ngram_rules": {
                    "1": {key1: nav.make_key(["كَلِمَة"])},
                    "2": {key2: nav.make_key(["كَلِمَة", "ثَانِيَة"])},
                },
            },
            {},
        )

        self.assertEqual(engine.diacritize_line("كلمة"), "كَلِمَة")
        self.assertEqual(
            engine.diacritize_line("كلمة ثانية"),
            "كَلِمَة ثَانِيَة",
        )

    def test_tashkeel_receives_text_without_existing_diacritics(self) -> None:
        inputs: list[str] = []

        def fake_tashkeel(text: str) -> str:
            inputs.append(text)
            return "اِنْعَطَفَ يَمِينًا."

        nav.tashkeel_run = fake_tashkeel
        engine = nav.LocalNavDiacritizer(
            {"line_rules": {}, "ngram_rules": {}},
            {},
        )

        self.assertEqual(
            engine.diacritize_line("اُنْعطف يمينا."),
            "اُنْعَطَفَ يَمِينًا.",
        )
        self.assertEqual(inputs, ["انعطف يمينا."])

    def test_tashkeel_result_is_not_cached(self) -> None:
        call_count = 0

        def fake_tashkeel(text: str) -> str:
            nonlocal call_count
            call_count += 1
            return "اِنْعَطَفَ."

        nav.tashkeel_run = fake_tashkeel
        engine = nav.LocalNavDiacritizer(
            {"line_rules": {}, "ngram_rules": {}},
            {},
        )

        engine.diacritize_line("انعطف.")
        engine.diacritize_line("انعطف.")
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()
