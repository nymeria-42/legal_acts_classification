#!/bin/bash

# Log file name
LOGFILE="process_log.txt"

{
  echo "Starting script at: $(date)";
  python process.py;
  python vocabulary.py;
  python classification_tfidf_within_agency.py;
  python classification_tfidf.py;
  python classification_bert.py;
  echo "Script finished at: $(date)"
} &> "$LOGFILE"