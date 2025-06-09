# Spiegazione delle Metriche di Trading

Questo documento fornisce una spiegazione completa delle metriche di trading specifiche utilizzate per valutare le prestazioni dei modelli di trading. Ogni metrica è descritta con il suo nome, una breve formula o metodo di calcolo (se applicabile), la sua interpretazione e il suo impatto sulla valutazione della performance di trading.

## Metriche di Performance Principali

### Total Return (Ritorno Totale)
*   **Nome Metrica:** Ritorno Totale
*   **Formula/Metodo di Calcolo:** `((Valore Finale Portafoglio - Valore Iniziale Portafoglio) / Valore Iniziale Portafoglio) * 100`
*   **Interpretazione:** Misura la percentuale di guadagno o perdita complessiva del portafoglio su un determinato periodo. Un valore più alto indica una performance migliore.
*   **Impatto sulla Valutazione:** Metrica fondamentale per comprendere la redditività grezza di una strategia.

### Sharpe Ratio
*   **Nome Metrica:** Sharpe Ratio
*   **Formula/Metodo di Calcolo:** `(Ritorno Medio Portafoglio - Tasso Risk-Free) / Deviazione Standard dei Ritorni`
*   **Interpretazione:** Misura il ritorno corretto per il rischio. Indica quanto rendimento extra è stato ottenuto per ogni unità di rischio (volatilità) assunta. Un valore più alto è preferibile.
*   **Impatto sulla Valutazione:** Aiuta a confrontare strategie con diversi livelli di rischio; una strategia con ritorni inferiori ma volatilità molto più bassa potrebbe avere uno Sharpe Ratio superiore.

### Sortino Ratio
*   **Nome Metrica:** Sortino Ratio
*   **Formula/Metodo di Calcolo:** `(Ritorno Medio Portafoglio - Tasso Minimo Accettabile di Ritorno) / Deviazione Standard dei Ritorni Negativi (Downside Deviation)`
*   **Interpretazione:** Simile allo Sharpe Ratio, ma considera solo la volatilità al ribasso (rischio di perdite). Un valore più alto indica una migliore performance corretta per il rischio di ribasso.
*   **Impatto sulla Valutazione:** Particolarmente utile per investitori avversi alle perdite, poiché penalizza solo la volatilità che porta a rendimenti inferiori a un target.

### Calmar Ratio
*   **Nome Metrica:** Calmar Ratio
*   **Formula/Metodo di Calcolo:** `Ritorno Annualizzato Composto / Massimo Drawdown`
*   **Interpretazione:** Misura il ritorno rispetto al massimo drawdown subito. Un valore più alto indica una migliore capacità di generare ritorni rispetto alla peggiore perdita storica.
*   **Impatto sulla Valutazione:** Fornisce una prospettiva sulla capacità di recupero di una strategia dopo periodi di perdita significativi.

## Metriche di Rischio

### Maximum Drawdown (Massimo Drawdown)
*   **Nome Metrica:** Massimo Drawdown
*   **Formula/Metodo di Calcolo:** La più grande perdita percentuale da un picco al successivo minimo durante un periodo specifico.
*   **Interpretazione:** Indica la peggiore perdita cumulativa che un investitore avrebbe subito se avesse investito al picco e venduto al minimo. Un valore più basso (più vicino a zero) è preferibile.
*   **Impatto sulla Valutazione:** Metrica cruciale per la gestione del rischio; drawdown elevati possono essere psicologicamente difficili da sostenere.

### Value at Risk (VaR) (Valore a Rischio)
*   **Nome Metrica:** Valore a Rischio (VaR)
*   **Formula/Metodo di Calcolo:** Stima statistica della massima perdita potenziale (in valore o percentuale) su un orizzonte temporale specifico con un dato livello di confidenza (es. 95% VaR).
*   **Interpretazione:** Ad esempio, un VaR del 5% a 1 giorno di €1000 significa che c'è una probabilità del 5% di perdere almeno €1000 nel giorno successivo. Valori più bassi indicano minor rischio.
*   **Impatto sulla Valutazione:** Aiuta a quantificare il rischio potenziale di ribasso in termini monetari o percentuali.

### Conditional VaR (CVaR) (VaR Condizionato)
*   **Nome Metrica:** VaR Condizionato (CVaR) o Expected Shortfall
*   **Formula/Metodo di Calcolo:** La perdita media attesa, dato che la perdita supera il livello del VaR.
*   **Interpretazione:** Fornisce una stima di "quanto male possono andare le cose" quando si verifica un evento di perdita estrema (oltre il VaR). Un CVaR più basso è preferibile.
*   **Impatto sulla Valutazione:** Offre una misura più conservativa del rischio di coda rispetto al VaR.

### Volatility (Volatilità)
*   **Nome Metrica:** Volatilità
*   **Formula/Metodo di Calcolo:** Deviazione standard dei ritorni su un periodo specifico.
*   **Interpretazione:** Misura la dispersione dei ritorni attorno alla media. Alta volatilità significa che i prezzi possono oscillare ampiamente, indicando un rischio maggiore. Bassa volatilità suggerisce stabilità.
*   **Impatto sulla Valutazione:** Componente chiave di molte metriche corrette per il rischio; indica l'incertezza o il rischio associato a una strategia.

## Metriche di Attività di Trading

### Win Rate (Tasso di Successo)
*   **Nome Metrica:** Tasso di Successo (Win Rate)
*   **Formula/Metodo di Calcolo:** `(Numero di Trade Profittevoli / Numero Totale di Trade) * 100`
*   **Interpretazione:** La percentuale di trade che hanno generato un profitto. Un win rate più alto è generalmente desiderabile, ma deve essere considerato insieme alla dimensione media delle vincite e delle perdite.
*   **Impatto sulla Valutazione:** Indica la frequenza con cui la strategia identifica opportunità profittevoli.

### Profit Factor (Fattore di Profitto)
*   **Nome Metrica:** Fattore di Profitto
*   **Formula/Metodo di Calcolo:** `Profitto Lordo Totale / Perdita Lorda Totale`
*   **Interpretazione:** Misura quanto profitto viene generato per ogni unità di perdita. Un valore superiore a 1 indica che la strategia è profittevole. Valori significativamente superiori a 1 sono preferibili.
*   **Impatto sulla Valutazione:** Fornisce una visione chiara della redditività complessiva, bilanciando vincite e perdite.

### Average Win (Vincita Media)
*   **Nome Metrica:** Vincita Media
*   **Formula/Metodo di Calcolo:** `Profitto Totale da Trade Vincenti / Numero di Trade Vincenti`
*   **Interpretazione:** L'importo medio guadagnato per ogni trade profittevole.
*   **Impatto sulla Valutazione:** Aiuta a capire l'entità dei guadagni quando la strategia ha successo.

### Average Loss (Perdita Media)
*   **Nome Metrica:** Perdita Media
*   **Formula/Metodo di Calcolo:** `Perdita Totale da Trade Perdenti / Numero di Trade Perdenti` (solitamente espresso come valore positivo)
*   **Interpretazione:** L'importo medio perso per ogni trade non profittevole.
*   **Impatto sulla Valutazione:** Aiuta a capire l'entità delle perdite quando la strategia fallisce. Il rapporto tra vincita media e perdita media è cruciale.

### Total Trades (Trade Totali)
*   **Nome Metrica:** Trade Totali
*   **Formula/Metodo di Calcolo:** Numero complessivo di operazioni di acquisto e vendita chiuse.
*   **Interpretazione:** Indica la frequenza di trading della strategia. Un numero elevato può implicare costi di transazione maggiori.
*   **Impatto sulla Valutazione:** Importante per valutare i costi di transazione e la robustezza statistica (più trade possono dare maggiore confidenza nelle altre metriche).

### Long Trades (Trade Long)
*   **Nome Metrica:** Trade Long
*   **Formula/Metodo di Calcolo:** Numero di operazioni in cui si è acquistato un asset con l'aspettativa che il suo prezzo aumenti.
*   **Interpretazione:** Indica l'attività della strategia nel prendere posizioni rialziste.
*   **Impatto sulla Valutazione:** Utile per analizzare la performance in diverse condizioni di mercato (es. mercati rialzisti).

### Short Trades (Trade Short)
*   **Nome Metrica:** Trade Short
*   **Formula/Metodo di Calcolo:** Numero di operazioni in cui si è venduto un asset (spesso preso in prestito) con l'aspettativa che il suo prezzo diminuisca, per poi riacquistarlo a un prezzo inferiore.
*   **Interpretazione:** Indica l'attività della strategia nel prendere posizioni ribassiste.
*   **Impatto sulla Valutazione:** Utile per analizzare la performance in diverse condizioni di mercato (es. mercati ribassisti).

## Metriche di Performance Avanzate

### Information Ratio
*   **Nome Metrica:** Information Ratio
*   **Formula/Metodo di Calcolo:** `(Ritorno Medio Portafoglio - Ritorno Medio Benchmark) / Deviazione Standard della Differenza tra Ritorni Portafoglio e Benchmark (Tracking Error)`
*   **Interpretazione:** Misura il rendimento extra generato rispetto a un benchmark, per unità di rischio assunto rispetto a quel benchmark. Un valore più alto indica una migliore capacità di sovraperformare attivamente il benchmark.
*   **Impatto sulla Valutazione:** Valuta l'abilità del gestore o della strategia nel generare alfa (sovraperformance).

### Treynor Ratio
*   **Nome Metrica:** Treynor Ratio
*   **Formula/Metodo di Calcolo:** `(Ritorno Medio Portafoglio - Tasso Risk-Free) / Beta del Portafoglio`
*   **Interpretazione:** Misura il ritorno extra ottenuto per ogni unità di rischio sistematico (rischio di mercato, misurato dal Beta). Un valore più alto è preferibile.
*   **Impatto sulla Valutazione:** Utile per portafogli ben diversificati dove il rischio sistematico è la principale preoccupazione.

### Jensen's Alpha (Alfa di Jensen)
*   **Nome Metrica:** Alfa di Jensen
*   **Formula/Metodo di Calcolo:** `Ritorno Medio Portafoglio - (Tasso Risk-Free + Beta del Portafoglio * (Ritorno Medio Benchmark - Tasso Risk-Free))`
*   **Interpretazione:** Misura il ritorno medio di un portafoglio sopra o sotto quello previsto dal Capital Asset Pricing Model (CAPM), dato il beta del portafoglio e il ritorno medio del mercato. Un alfa positivo indica sovraperformance.
*   **Impatto sulla Valutazione:** Indica se una strategia ha generato ritorni superiori a quelli giustificati dal suo livello di rischio sistematico.

### Beta
*   **Nome Metrica:** Beta
*   **Formula/Metodo di Calcolo:** Covarianza tra i ritorni del portafoglio e i ritorni del benchmark, divisa per la varianza dei ritorni del benchmark.
*   **Interpretazione:** Misura la volatilità di un portafoglio rispetto al mercato o a un benchmark. Un Beta > 1 indica che il portafoglio è più volatile del mercato; Beta < 1 indica meno volatilità; Beta = 1 indica volatilità pari al mercato.
*   **Impatto sulla Valutazione:** Aiuta a comprendere come una strategia potrebbe comportarsi in relazione ai movimenti generali del mercato.

### R-squared (R Quadrato)
*   **Nome Metrica:** R Quadrato
*   **Formula/Metodo di Calcolo:** La percentuale della variazione dei ritorni di un portafoglio che è spiegata dalla variazione dei ritorni del benchmark.
*   **Interpretazione:** Valori vicini al 100% indicano che i movimenti del portafoglio sono altamente correlati a quelli del benchmark. Valori bassi suggeriscono che altri fattori influenzano i ritorni del portafoglio.
*   **Impatto sulla Valutazione:** Indica quanto fedelmente una strategia segue un benchmark e quanto della sua performance è attribuibile ai movimenti del benchmark.

## Analisi del Drawdown

### Current Drawdown (Drawdown Attuale)
*   **Nome Metrica:** Drawdown Attuale
*   **Formula/Metodo di Calcolo:** La percentuale di calo dal picco più recente del valore del portafoglio al valore attuale.
*   **Interpretazione:** Indica quanto il portafoglio è attualmente al di sotto del suo massimo storico recente.
*   **Impatto sulla Valutazione:** Fornisce una visione in tempo reale della performance rispetto ai picchi recenti.

### Drawdown Duration (Durata del Drawdown)
*   **Nome Metrica:** Durata del Drawdown
*   **Formula/Metodo di Calcolo:** Il periodo di tempo dall'inizio di un drawdown (quando il portafoglio scende da un picco) fino a quando il portafoglio recupera quel picco precedente.
*   **Interpretazione:** Misura per quanto tempo un investitore dovrebbe attendere prima che una perdita venga recuperata. Durate più brevi sono preferibili.
*   **Impatto sulla Valutazione:** Importante per la tolleranza al rischio e per comprendere la resilienza di una strategia.

### Recovery Factor (Fattore di Recupero)
*   **Nome Metrica:** Fattore di Recupero
*   **Formula/Metodo di Calcolo:** `Profitto Netto Totale / Massimo Drawdown`
*   **Interpretazione:** Misura la capacità di una strategia di generare profitti rispetto alla sua peggiore perdita storica. Un valore più alto indica una maggiore capacità di recuperare dalle perdite.
*   **Impatto sulla Valutazione:** Valuta l'efficienza con cui una strategia recupera le perdite e continua a generare profitti.