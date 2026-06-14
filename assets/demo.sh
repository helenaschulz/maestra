#!/usr/bin/env bash
# Demo player for the README terminal GIF (assets/demo.tape).
#
# Defines a `maestra` shell function that replays the *real* output of a Maestra run on
# Titanic — the cleaning plan, applied drops/imputations and holdout metrics are taken
# verbatim from an actual run; only AutoGluon's verbose fit logs and the training wait are
# trimmed so the GIF stays short and readable. Numbers are real.

maestra() {
  local C='\033[36m' G='\033[32m' Y='\033[33m' D='\033[2m' B='\033[1m' R='\033[0m'
  echo -e "${D}Loaded data/titanic.csv: rows=891, columns=12${R}"
  sleep 0.9

  echo -e "\n${C}${B}=== LLM cleaning plan (gpt-4o) ===${R}"
  sleep 0.5
  echo -e "${D}{ columns_to_drop: [PassengerId, Name, Ticket, Cabin],"
  echo -e "  imputations:     [Age → median, Embarked → most_frequent] }${R}"
  sleep 1.1

  echo -e "\n${C}${B}=== Applied ===${R}"
  sleep 0.3
  local lines=(
    "DROP 'PassengerId'  -- ID-like, unique per row"
    "DROP 'Name'         -- high-cardinality free text"
    "DROP 'Cabin'        -- 77% missing"
    "IMPUTE 'Age' [median] fit on train (missing=140) -> 28.0"
    "IMPUTE 'Embarked' [most_frequent] -> 'S'"
  )
  for l in "${lines[@]}"; do echo -e "  ${G}${l}${R}"; sleep 0.45; done
  echo -e "Columns after cleaning: ${Y}${B}8${R} (from 12)"
  sleep 1.0

  echo -e "\n${C}${B}=== Best-model metrics on holdout ===${R}"
  sleep 0.4
  echo -e "  accuracy: ${Y}${B}0.826${R}"
  echo -e "  roc_auc:  ${Y}${B}0.884${R}"
  sleep 0.4
}
