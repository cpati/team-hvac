### team-hvac

#### Alternus Vera
**Dataset: Politifact/Liar-Liar dataset (https://www.politifact.com)**

Description of the train TSV file format:
- Column 1: the ID of the statement ([ID].json)
- Column 2: the label.
- Column 3: the statement.
- Column 4: the subject(s).
- Column 5: the speaker.
- Column 6: the speaker's job title.
- Column 7: the state info.
- Column 8: the party affiliation.
- Column 9-13: the total credit history count, including the current statement. 
  - 9: barely true counts.
  - 10: false counts.
  - 11: half true counts.
  - 12: mostly true counts.
  - 13: pants on fire counts.
- Column 14: the context (venue / location of the speech or statement).

#### Summary of the considered factors:
* Political Affiliation: How much any political news is inclined towards a respective party
* Sentiment Analysis: Sentiments of people can result in biasing the news
* Sensationalism: Extreme sensationalism is an indication of fakeness. This derieved on the basis of Setiments
* Stance detection: Many versions of the same news are published in several traditional and new media. The language of these articles vary from supporting or denying a claim
* Spam Detection: Spam or Ham are untrue articles/texts made viral in order to deceive masses
* Spelling Error: More spelling errors mean news from unreliable sources

##### Alternus Vera Score:
Fakeness = 0.9 * (Sensationalism) + 0.85 * (Political Affiliation) + 0.65 * (Spam Detection) + 0.75  * (Stance Detection) + 0.8 * (Spelling Error)	
