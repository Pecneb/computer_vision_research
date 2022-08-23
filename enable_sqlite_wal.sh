ls research_data/*/*.db | while read filename;
  do sqlite3 $filename 'PRAGMA journal_mode=WAL;';
done;
