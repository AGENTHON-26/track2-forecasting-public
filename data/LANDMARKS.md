# Landmarks index

Some practice units ship one or more **landmark documents**: a single official statement, speech,
testimony or set of minutes that the card's text corpus is built around. In each unit's
`text/corpus_index.json` these entries have `doc_type = "landmark"` and the source
`official (see landmarks index)`, and the unit's `card.toml` names that source in `[text]`. This
file is that index.

The track ships 13 distinct landmark documents, as 27 copies across 24 units. For each one the
table gives the document's own date, its title as printed on the document, and the official
source URL recorded by the organizers' landmark fetcher. The retrieval date was not recorded.

| File (under `units/<unit>/text/`) | Document date | Title on the document | Official source |
|---|---|---|---|
| `draghi_whatever_it_takes_2012.txt` | 2012-07-26 | Speech by Mario Draghi at the Global Investment Conference in London | <https://www.ecb.europa.eu/press/key/date/2012/html/sp120726.en.html> |
| `bernanke_jec_testimony_2013.txt` | 2013-05-22 | Bernanke, "The Economic Outlook", before the Joint Economic Committee | <https://www.federalreserve.gov/newsevents/testimony/bernanke20130522a.htm> |
| `fomc_minutes_20130619_released_2013-07-10.txt` | meeting 2013-06-19, released 2013-07-10 | Minutes of the Federal Open Market Committee, June 18-19, 2013 | <https://www.federalreserve.gov/monetarypolicy/fomcminutes20130619.htm> |
| `snb_floor_discontinued_20150115.txt` | 2015-01-15 | Swiss National Bank discontinues minimum exchange rate and lowers interest rate to –0.75% (press release) | <https://www.snb.ch/public/asset/en/www-snb-ch/publications/communication/press-releases/2015/pre_20150115/publications0_en/pre_20150115.en.pdf> |
| `fomc_statement_20200315.txt` | 2020-03-15 | Federal Reserve issues FOMC statement | <https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a.htm> |
| `fomc_statement_20210922.txt` | 2021-09-22 | Federal Reserve issues FOMC statement | <https://www.federalreserve.gov/newsevents/pressreleases/monetary20210922a.htm> |
| `fomc_statement_20211215.txt` | 2021-12-15 | Federal Reserve issues FOMC statement | <https://www.federalreserve.gov/newsevents/pressreleases/monetary20211215a.htm> |
| `powell_jackson_hole_2022.txt` | 2022-08-26 | Powell, "Monetary Policy and Price Stability" (Jackson Hole symposium) | <https://www.federalreserve.gov/newsevents/speech/powell20220826a.htm> |
| `boe_mpc_statement_20220922.txt` | 2022-09-22 | Bank Rate increased to 2.25% - September 2022 Monetary Policy Summary and minutes of the Monetary Policy Committee meeting | <https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2022/september-2022> |
| `boj_ycc_20221220.txt` | 2022-12-20 | Statement on Monetary Policy | <https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2022/k221220a.pdf> |
| `boj_ycc_20230728.txt` | 2023-07-28 | Statement on Monetary Policy | <https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2023/k230728a.pdf> |
| `boj_ycc_20240319.txt` | 2024-03-19 | Changes in the Monetary Policy Framework | <https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2024/k240319a.pdf> |
| `boj_ycc_20240731.txt` | 2024-07-31 | Change in the Guideline for Money Market Operations and Decision on the Plan for the Reduction of the Purchase Amount of Japanese Government Bonds | <https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2024/k240731a.pdf> |

The rights position of each document is in its manifest entry and in
[`THIRD-PARTY-NOTICES.md`](../THIRD-PARTY-NOTICES.md): the Federal Reserve documents are U.S.
Government works in the public domain; the ECB speech prints its own reproduction permission; for
the Bank of England, Bank of Japan and Swiss National Bank documents the issuer's terms govern
(for the SNB press release, its copyright page, <https://www.snb.ch/en/srv/disclaimer_copyright>,
allows non-commercial use compatible with the purpose of the information).

Every copy has been passed through `scripts/declutter_corpus.py` like the rest of the corpus (see
`data/PROVENANCE.md`); only the ECB and Bank of England pages had website material to remove.
`boe_mpc_statement_20220922.txt` no longer carries the "Other Monetary
Policy Committee news" list that the web page showed at retrieval time; that list named 2026
announcements, years after the statement.
