# Spiegazione delle Metriche di Trading

Questo documento fornisce una spiegazione completa delle metriche di trading specifiche utilizzate per valutare le prestazioni dei modelli di trading. Ogni metrica è descritta con il suo nome, una breve formula o metodo di calcolo (se applicabile), la sua interpretazione e il suo impatto sulla valutazione della performance di trading.

## Metriche di Performance Principali

### Total Return (Ritorno Totale)
*   **Nome Metrica:** Ritorno Totale
*   **Formula/Metodo di Calcolo:** `((Valore Finale Portafoglio - Valore Iniziale Portafoglio) / Valore Iniziale Portafoglio) * 100`
*   **Interpretazione:** Misura la percentuale di guadagno o perdita complessiva del portafoglio su un determinato periodo.
    *   **Valori Positivi (>0%):** Indicano un profitto. Un valore significativamente alto (es. >20-30% annuo, a seconda del contesto) è generalmente considerato molto buono, ma deve essere valutato in relazione al rischio assunto.
    *   **Valori Negativi (<0%):** Indicano una perdita.
    *   **Zero (0%):** Nessun guadagno né perdita (break-even).
    *   È importante contestualizzare il ritorno rispetto agli obiettivi, al periodo di tempo e al benchmark di riferimento.
*   **Impatto sulla Valutazione:** Metrica fondamentale e punto di partenza per comprendere la redditività grezza di una strategia. Tuttavia, da sola non fornisce un quadro completo, poiché non considera il rischio assunto per ottenere tale ritorno, né la consistenza della performance nel tempo. Va analizzata insieme ad altre metriche come Sharpe Ratio e Maximum Drawdown.

### Sharpe Ratio
*   **Nome Metrica:** Sharpe Ratio
*   **Formula/Metodo di Calcolo:** `(Ritorno Medio Portafoglio - Tasso Risk-Free) / Deviazione Standard dei Ritorni`
*   **Interpretazione:** Misura il ritorno corretto per il rischio, specificamente per la volatilità totale. Indica quanto rendimento extra è stato ottenuto per ogni unità di rischio (volatilità) assunta.
    *   **<1:** Generalmente considerato subottimale o scarso, poiché il ritorno non compensa adeguatamente il rischio.
    *   **1-1.99:** Considerato accettabile o buono.
    *   **2-2.99:** Considerato molto buono.
    *   **>3:** Considerato eccellente.
    *   Questi intervalli sono indicativi e possono variare a seconda della classe di asset e delle condizioni di mercato. Un valore più alto è sempre preferibile.
*   **Impatto sulla Valutazione:** Essenziale per confrontare strategie con diversi livelli di rischio e rendimento. Una strategia con ritorni nominali inferiori ma volatilità molto più bassa potrebbe avere uno Sharpe Ratio superiore, indicando una migliore qualità del ritorno per unità di rischio. Aiuta a identificare strategie efficienti.

### Sortino Ratio
*   **Nome Metrica:** Sortino Ratio
*   **Formula/Metodo di Calcolo:** `(Ritorno Medio Portafoglio - Tasso Minimo Accettabile di Ritorno) / Deviazione Standard dei Ritorni Negativi (Downside Deviation)`
*   **Interpretazione:** Simile allo Sharpe Ratio, ma considera solo la volatilità al ribasso (rischio di perdite), ovvero la deviazione standard dei ritorni negativi. Un valore più alto indica una migliore performance corretta per il rischio di ribasso.
    *   Come per lo Sharpe Ratio, valori più alti sono migliori. Un Sortino Ratio > 2 è spesso considerato buono.
    *   Se il Sortino Ratio è significativamente più alto dello Sharpe Ratio, suggerisce che gran parte della volatilità della strategia è al rialzo (positiva) e non al ribasso.
*   **Impatto sulla Valutazione:** Particolarmente utile per investitori avversi alle perdite, poiché penalizza solo la volatilità che porta a rendimenti inferiori a un target (Tasso Minimo Accettabile di Ritorno, MAR). Fornisce una misura più raffinata del rendimento corretto per il rischio per chi è preoccupato principalmente dalle perdite.

### Calmar Ratio
*   **Nome Metrica:** Calmar Ratio
*   **Formula/Metodo di Calcolo:** `Ritorno Annualizzato Composto / Massimo Drawdown` (il drawdown è espresso come valore positivo)
*   **Interpretazione:** Misura il ritorno rispetto al massimo drawdown subito. Un valore più alto indica una migliore capacità di generare ritorni rispetto alla peggiore perdita storica.
    *   **<0.5:** Generalmente considerato scarso.
    *   **0.5-1.0:** Accettabile.
    *   **>1.0:** Buono.
    *   **>3.0:** Eccellente, ma dipende dal tipo di strategia e dal periodo di osservazione (tipicamente calcolato su almeno 3 anni).
*   **Impatto sulla Valutazione:** Fornisce una prospettiva sulla capacità di recupero di una strategia dopo periodi di perdita significativi. È particolarmente rilevante per strategie a lungo termine e per valutare la "dolorosità" del percorso di investimento.

## Metriche di Rischio

### Maximum Drawdown (Massimo Drawdown)
*   **Nome Metrica:** Massimo Drawdown
*   **Formula/Metodo di Calcolo:** La più grande perdita percentuale da un picco al successivo minimo durante un periodo specifico.
*   **Interpretazione:** Indica la peggiore perdita cumulativa che un investitore avrebbe subito se avesse investito al picco e venduto al minimo. Un valore più basso (più vicino a zero) è preferibile.
    *   **<10%:** Spesso considerato basso e accettabile per strategie conservative.
    *   **10-25%:** Moderato, può essere accettabile per strategie bilanciate.
    *   **25-50%:** Alto, tipico di strategie più aggressive o in mercati volatili.
    *   **>50%:** Molto alto, indica un rischio significativo di perdite ingenti.
    *   La tolleranza al drawdown è soggettiva e dipende dagli obiettivi dell'investitore.
*   **Impatto sulla Valutazione:** Metrica cruciale per la gestione del rischio; drawdown elevati possono essere psicologicamente difficili da sostenere e possono portare all'abbandono della strategia. Indica il rischio di "worst-case scenario" sperimentato storicamente.

### Value at Risk (VaR) (Valore a Rischio)
*   **Nome Metrica:** Valore a Rischio (VaR)
*   **Formula/Metodo di Calcolo:** Stima statistica della massima perdita potenziale (in valore o percentuale) su un orizzonte temporale specifico con un dato livello di confidenza (es. 95% VaR).
*   **Interpretazione:** Ad esempio, un VaR del 5% a 1 giorno di €1000 (o 1% del portafoglio) significa che c'è una probabilità del 5% di perdere almeno €1000 (o 1% del portafoglio) nel giorno successivo, assumendo condizioni di mercato normali. Valori più bassi (in termini assoluti) indicano minor rischio per un dato livello di confidenza.
*   **Impatto sulla Valutazione:** Aiuta a quantificare il rischio potenziale di ribasso in termini monetari o percentuali, fornendo un limite di perdita attesa in condizioni normali. Utilizzato per la definizione di limiti di rischio e per l'allocazione del capitale.

### Conditional VaR (CVaR) (VaR Condizionato)
*   **Nome Metrica:** VaR Condizionato (CVaR) o Expected Shortfall
*   **Formula/Metodo di Calcolo:** La perdita media attesa, dato che la perdita supera il livello del VaR.
*   **Interpretazione:** Fornisce una stima di "quanto male possono andare le cose" quando si verifica un evento di perdita estrema (cioè, quando la perdita supera il VaR). Ad esempio, se il VaR al 95% è del 2% e il CVaR al 95% è del 3.5%, significa che nel 5% dei casi peggiori, la perdita media attesa è del 3.5%. Un CVaR più basso è preferibile. È sempre maggiore o uguale al VaR (per lo stesso livello di confidenza).
*   **Impatto sulla Valutazione:** Offre una misura più conservativa e completa del rischio di coda (tail risk) rispetto al VaR, poiché considera l'entità delle perdite estreme, non solo la loro soglia.

### Volatility (Volatilità)
*   **Nome Metrica:** Volatilità
*   **Formula/Metodo di Calcolo:** Deviazione standard dei ritorni su un periodo specifico (spesso annualizzata).
*   **Interpretazione:** Misura la dispersione dei ritorni attorno alla media.
    *   **Alta volatilità:** I prezzi/ritorni possono oscillare ampiamente, indicando un rischio e un'incertezza maggiori.
    *   **Bassa volatilità:** Suggerisce maggiore stabilità e prevedibilità dei ritorni.
    *   Non esiste un valore "buono" o "cattivo" in assoluto; dipende dal tipo di strategia (es. strategie trend-following potrebbero accettare/sfruttare alta volatilità, mentre strategie market-neutral cercano bassa volatilità) e dalla tolleranza al rischio dell'investitore. Va confrontata con la volatilità del benchmark o storica.
*   **Impatto sulla Valutazione:** Componente chiave di molte metriche corrette per il rischio (come lo Sharpe Ratio). Indica l'incertezza o il rischio associato a una strategia. Una volatilità elevata può rendere difficile mantenere la strategia a causa delle oscillazioni di valore.

## Metriche di Attività di Trading

### Win Rate (Tasso di Successo)
*   **Nome Metrica:** Tasso di Successo (Win Rate)
*   **Formula/Metodo di Calcolo:** `(Numero di Trade Profittevoli / Numero Totale di Trade) * 100`
*   **Interpretazione:** La percentuale di trade che hanno generato un profitto.
    *   Un win rate > 50% è spesso percepito come buono, ma non è sufficiente da solo. Ad esempio, un win rate del 40% può essere molto profittevole se le vincite medie sono significativamente maggiori delle perdite medie. Viceversa, un win rate del 70% può essere perdente se le perdite medie sono molto più grandi delle vincite medie.
*   **Impatto sulla Valutazione:** Indica la frequenza con cui la strategia identifica opportunità profittevoli. Un win rate elevato può aumentare la fiducia nella strategia, ma deve essere bilanciato con il rapporto tra vincita media e perdita media (Risk/Reward Ratio).

### Profit Factor (Fattore di Profitto)
*   **Nome Metrica:** Fattore di Profitto
*   **Formula/Metodo di Calcolo:** `Profitto Lordo Totale / Valore Assoluto della Perdita Lorda Totale`
*   **Interpretazione:** Misura quanto profitto viene generato per ogni unità di perdita.
    *   **<1:** La strategia è perdente.
    *   **=1:** Break-even (ignorando i costi).
    *   **1-1.5:** Marginalmente profittevole, potrebbe non coprire i costi.
    *   **1.5-2.0:** Considerato buono.
    *   **>2.0:** Considerato molto buono/eccellente.
*   **Impatto sulla Valutazione:** Fornisce una visione chiara della redditività complessiva, bilanciando l'ammontare totale delle vincite e delle perdite. È una misura robusta della capacità di generare profitto.

### Average Win (Vincita Media)
*   **Nome Metrica:** Vincita Media
*   **Formula/Metodo di Calcolo:** `Profitto Totale da Trade Vincenti / Numero di Trade Vincenti`
*   **Interpretazione:** L'importo medio guadagnato per ogni trade profittevole. Un valore più alto è generalmente migliore, ma va contestualizzato con il win rate e la perdita media.
*   **Impatto sulla Valutazione:** Aiuta a capire l'entità dei guadagni quando la strategia ha successo. Insieme alla perdita media e al win rate, determina l'aspettativa matematica della strategia.

### Average Loss (Perdita Media)
*   **Nome Metrica:** Perdita Media
*   **Formula/Metodo di Calcolo:** `Valore Assoluto della Perdita Totale da Trade Perdenti / Numero di Trade Perdenti`
*   **Interpretazione:** L'importo medio perso per ogni trade non profittevole. Un valore più basso è generalmente migliore. È cruciale confrontarlo con la vincita media.
*   **Impatto sulla Valutazione:** Aiuta a capire l'entità delle perdite quando la strategia fallisce. Un obiettivo comune è avere una vincita media superiore alla perdita media, specialmente se il win rate non è estremamente alto.

### Total Trades (Trade Totali)
*   **Nome Metrica:** Trade Totali
*   **Formula/Metodo di Calcolo:** Numero complessivo di operazioni di acquisto e vendita chiuse.
*   **Interpretazione:** Indica la frequenza di trading della strategia. Non c'è un valore "ideale" universale; dipende dalla strategia.
    *   **Alto numero:** Tipico di strategie high-frequency o scalping. Può implicare costi di transazione significativi e richiedere infrastrutture robuste.
    *   **Basso numero:** Tipico di strategie a lungo termine o swing trading. Può rendere più difficile ottenere significatività statistica per le altre metriche.
*   **Impatto sulla Valutazione:** Importante per valutare l'impatto dei costi di transazione (commissioni, slippage). Un numero sufficientemente elevato di trade è necessario per avere fiducia nella validità statistica delle altre metriche di performance.

### Long Trades (Trade Long)
*   **Nome Metrica:** Trade Long
*   **Formula/Metodo di Calcolo:** Numero di operazioni in cui si è acquistato un asset con l'aspettativa che il suo prezzo aumenti.
*   **Interpretazione:** Indica l'attività della strategia nel prendere posizioni rialziste. La proporzione di trade long rispetto al totale può indicare un bias direzionale.
*   **Impatto sulla Valutazione:** Utile per analizzare la performance in diverse condizioni di mercato (es. mercati rialzisti vs. ribassisti) e per capire se la strategia è specializzata o adattabile.

### Short Trades (Trade Short)
*   **Nome Metrica:** Trade Short
*   **Formula/Metodo di Calcolo:** Numero di operazioni in cui si è venduto un asset (spesso preso in prestito) con l'aspettativa che il suo prezzo diminuisca, per poi riacquistarlo a un prezzo inferiore.
*   **Interpretazione:** Indica l'attività della strategia nel prendere posizioni ribassiste.
*   **Impatto sulla Valutazione:** Simile ai trade long, aiuta a capire la performance in mercati ribassisti e il bias della strategia. Alcune strategie potrebbero non effettuare short trade.

## Metriche di Performance Avanzate

### Information Ratio
*   **Nome Metrica:** Information Ratio
*   **Formula/Metodo di Calcolo:** `(Ritorno Medio Portafoglio - Ritorno Medio Benchmark) / Deviazione Standard della Differenza tra Ritorni Portafoglio e Benchmark (Tracking Error)`
*   **Interpretazione:** Misura il rendimento extra (alfa) generato rispetto a un benchmark, per unità di rischio assunto rispetto a quel benchmark (tracking error).
    *   **>0:** La strategia ha sovraperformato il benchmark su base corretta per il rischio relativo.
    *   **>0.5:** Spesso considerato buono.
    *   **>1.0:** Spesso considerato molto buono, indicando una forte e consistente sovraperformance.
    *   Un valore più alto indica una migliore capacità di sovraperformare attivamente e consistentemente il benchmark.
*   **Impatto sulla Valutazione:** Valuta l'abilità del gestore o della strategia nel generare alfa (sovraperformance) in modo consistente. È una metrica chiave per la gestione attiva.

### Treynor Ratio
*   **Nome Metrica:** Treynor Ratio
*   **Formula/Metodo di Calcolo:** `(Ritorno Medio Portafoglio - Tasso Risk-Free) / Beta del Portafoglio`
*   **Interpretazione:** Misura il ritorno extra ottenuto (oltre il tasso risk-free) per ogni unità di rischio sistematico (rischio di mercato, misurato dal Beta). Un valore più alto è preferibile, indicando un miglior compenso per l'assunzione di rischio di mercato.
*   **Impatto sulla Valutazione:** Utile per portafogli ben diversificati dove il rischio sistematico (beta) è la principale preoccupazione. Confronta i ritorni con il solo rischio di mercato, ignorando il rischio specifico.

### Jensen's Alpha (Alfa di Jensen)
*   **Nome Metrica:** Alfa di Jensen
*   **Formula/Metodo di Calcolo:** `Ritorno Medio Portafoglio - (Tasso Risk-Free + Beta del Portafoglio * (Ritorno Medio Benchmark - Tasso Risk-Free))`
*   **Interpretazione:** Misura il ritorno medio di un portafoglio sopra o sotto quello previsto dal Capital Asset Pricing Model (CAPM), dato il beta del portafoglio e il ritorno medio del mercato.
    *   **Alfa > 0:** La strategia ha sovraperformato le attese basate sul suo rischio sistematico.
    *   **Alfa < 0:** La strategia ha sottoperformato.
    *   **Alfa = 0:** La strategia ha performato in linea con le attese.
    *   Un alfa positivo e statisticamente significativo è l'obiettivo della gestione attiva.
*   **Impatto sulla Valutazione:** Indica se una strategia ha generato ritorni superiori (o inferiori) a quelli giustificati dal suo livello di rischio sistematico. È una misura diretta della "value added" dal gestore.

### Beta
*   **Nome Metrica:** Beta
*   **Formula/Metodo di Calcolo:** Covarianza tra i ritorni del portafoglio e i ritorni del benchmark, divisa per la varianza dei ritorni del benchmark.
*   **Interpretazione:** Misura la volatilità di un portafoglio rispetto al mercato o a un benchmark.
    *   **Beta = 1:** Il portafoglio tende a muoversi in linea con il mercato.
    *   **Beta > 1:** Il portafoglio è più volatile del mercato (es. Beta 1.2 significa il 20% più volatile).
    *   **Beta < 1 (ma > 0):** Il portafoglio è meno volatile del mercato.
    *   **Beta = 0:** Nessuna correlazione con i movimenti del mercato.
    *   **Beta < 0:** Il portafoglio tende a muoversi in direzione opposta al mercato (raro per portafogli diversificati).
*   **Impatto sulla Valutazione:** Aiuta a comprendere come una strategia potrebbe comportarsi in relazione ai movimenti generali del mercato e il suo contributo al rischio sistematico di un portafoglio più ampio.

### R-squared (R Quadrato)
*   **Nome Metrica:** R Quadrato
*   **Formula/Metodo di Calcolo:** La percentuale della variazione dei ritorni di un portafoglio che è spiegata dalla variazione dei ritorni del benchmark. Valori da 0 a 1 (o 0% a 100%).
*   **Interpretazione:**
    *   Valori vicini al 100% (es. > 85-90%): Indicano che i movimenti del portafoglio sono altamente correlati e spiegati da quelli del benchmark. In questo caso, Beta e Alfa sono metriche più significative.
    *   Valori bassi (es. < 70%): Suggeriscono che altri fattori, non correlati al benchmark, influenzano significativamente i ritorni del portafoglio. In questo caso, Beta e Alfa potrebbero essere meno rappresentativi.
*   **Impatto sulla Valutazione:** Indica quanto fedelmente una strategia segue un benchmark e quanto della sua performance è attribuibile ai movimenti del benchmark. Un R-quadrato basso può significare che la strategia è veramente unica o che il benchmark scelto non è appropriato.

## Analisi del Drawdown

### Current Drawdown (Drawdown Attuale)
*   **Nome Metrica:** Drawdown Attuale
*   **Formula/Metodo di Calcolo:** La percentuale di calo dal picco più recente del valore del portafoglio al valore attuale.
*   **Interpretazione:** Indica quanto il portafoglio è attualmente al di sotto del suo massimo storico recente. Un valore di 0% significa che il portafoglio è a un nuovo picco (o uguale al picco precedente). Un valore negativo (es. -5%) indica che il portafoglio è il 5% sotto il suo ultimo picco.
*   **Impatto sulla Valutazione:** Fornisce una visione in tempo reale della performance rispetto ai picchi recenti e può essere un segnale di allerta se il drawdown attuale si avvicina o supera livelli di tolleranza.

### Drawdown Duration (Durata del Drawdown)
*   **Nome Metrica:** Durata del Drawdown
*   **Formula/Metodo di Calcolo:** Il periodo di tempo dall'inizio di un drawdown (quando il portafoglio scende da un picco) fino a quando il portafoglio recupera quel picco precedente.
*   **Interpretazione:** Misura per quanto tempo un investitore dovrebbe attendere prima che una perdita venga recuperata. Durate più brevi sono preferibili. Lunghe durate possono essere psicologicamente difficili da sopportare.
*   **Impatto sulla Valutazione:** Importante per la tolleranza al rischio e per comprendere la resilienza di una strategia. Strategie con drawdown lunghi, anche se poi recuperano, possono non essere adatte a tutti gli investitori.

### Recovery Factor (Fattore di Recupero)
*   **Nome Metrica:** Fattore di Recupero
*   **Formula/Metodo di Calcolo:** `Profitto Netto Totale / Valore Assoluto del Massimo Drawdown`
*   **Interpretazione:** Misura la capacità di una strategia di generare profitti rispetto alla sua peggiore perdita storica.
    *   **<1:** La strategia non ha recuperato il suo massimo drawdown.
    *   **>1:** La strategia ha generato più profitti del suo massimo drawdown. Valori più alti sono migliori.
    *   Un valore di 2.0, ad esempio, significa che la strategia ha guadagnato il doppio del suo peggior drawdown.
*   **Impatto sulla Valutazione:** Valuta l'efficienza con cui una strategia recupera le perdite e continua a generare profitti. Un alto fattore di recupero è indice di una strategia robusta e resiliente.

## Metriche Specifiche di Reinforcement Learning

Questa sezione descrive le metriche comunemente utilizzate per valutare le prestazioni e il processo di apprendimento degli agenti di Reinforcement Learning (RL), con particolare attenzione alle loro applicazioni e interpretazioni nel contesto del trading e specificamente per modelli come DQN (Deep Q-Network).

### Episode Reward (Ricompensa dell'Episodio)
*   **Nome Metrica:** Ricompensa dell'Episodio (Total Reward per Episode)
*   **Cosa mostra:** La somma totale delle ricompense ottenute dall'agente in un singolo episodio di training o valutazione. Nel trading, la ricompensa è spesso legata al profitto/perdita o a metriche finanziarie modificate.
*   **Importanza:** Metrica fondamentale in RL, indica direttamente quanto bene l'agente sta imparando a raggiungere l'obiettivo definito dalla funzione di ricompensa (es. massimizzare i profitti, minimizzare i rischi).
*   **Interpretazione:**
    *   *Trend Crescente:* Generalmente indica che l'agente sta migliorando le sue decisioni e la sua policy nel tempo.
    *   *Stabilità/Plateau:* Può indicare che l'agente ha raggiunto una performance ottimale (o un ottimo locale) o che l'apprendimento è rallentato.
    *   *Alta Varianza:* Fluttuazioni significative tra episodi possono indicare instabilità nell'apprendimento, un ambiente molto stocastico, o un'esplorazione eccessiva.
    *   *Valori Negativi/Bassi:* L'agente sta fallendo nel raggiungere l'obiettivo o sta subendo penalità.
    *   *Per DQN nel trading:* Una ricompensa positiva potrebbe correlarsi a un profitto nell'episodio, una negativa a una perdita. La scala e la natura della ricompensa (es. P&L diretto, Sharpe Ratio differenziale) influenzano direttamente l'interpretazione.

### Episode Length (Durata dell'Episodio)
*   **Nome Metrica:** Durata dell'Episodio
*   **Cosa mostra:** Il numero di passi temporali (time steps) o decisioni prese dall'agente prima che un episodio termini.
*   **Importanza:** Può indicare l'efficienza dell'agente, la sua capacità di sopravvivere in un ambiente (es. non esaurire il capitale), o la velocità nel raggiungere un obiettivo.
*   **Interpretazione:**
    *   *Compiti Finiti (es. raggiungere un target di profitto):* Una durata più breve potrebbe essere desiderabile se l'obiettivo è raggiunto rapidamente.
    *   *Compiti Continui/Sopravvivenza (es. trading per un periodo fisso):* La durata potrebbe essere fissa o, se variabile (es. termina per stop loss), una durata più lunga con ricompensa positiva è generalmente migliore.
    *   *Correlazione con Ricompensa:* Analizzare sempre insieme alla ricompensa. Una lunga durata con bassa o negativa ricompensa non è ideale.
    *   *Per DQN nel trading:* Potrebbe rappresentare il numero di barre di prezzo o decisioni di trading prima che l'episodio finisca (es. per esaurimento capitale, raggiungimento di un limite temporale, o fine del dataset di backtest).

### Loss (Perdita del Modello)
*   **Nome Metrica:** Perdita del Modello (es. Perdita della Funzione di Valore/Q-value per DQN)
*   **Cosa mostra:** Una misura di quanto le predizioni del modello (es. Q-values in DQN) differiscono dai valori target. Indica l'errore del modello durante l'apprendimento.
*   **Importanza:** Cruciale per monitorare il processo di training. Una perdita che non diminuisce o che diverge può indicare problemi di apprendimento (es. learning rate errato, instabilità numerica, cattiva definizione della ricompensa).
*   **Interpretazione:**
    *   *Trend Decrescente:* Idealmente, la perdita dovrebbe diminuire nel tempo, indicando che il modello sta convergendo e le sue stime stanno diventando più accurate.
    *   *Stabilità/Plateau Basso:* Può indicare una buona convergenza.
    *   *Valori Alti/Crescenti:* Segnali di problemi di apprendimento.
    *   *Fluttuazioni Eccessive:* Possono indicare instabilità.
    *   *Per DQN:* Tipicamente la Mean Squared Error (MSE) tra i Q-values predetti e i target Q-values (calcolati con l'equazione di Bellman). Una loss che converge a zero non è sempre l'obiettivo (specialmente in ambienti stocastici), ma dovrebbe stabilizzarsi.

### Entropy (Entropia della Policy / Epsilon per DQN)
*   **Nome Metrica:** Entropia (o proxy come Epsilon per strategie epsilon-greedy come DQN)
*   **Cosa mostra:** Misura la casualità o l'incertezza nelle azioni scelte dall'agente. Un'alta entropia significa più esplorazione (azioni più casuali), una bassa entropia significa più sfruttamento (azioni più deterministiche basate sulla policy appresa).
*   **Importanza:** Bilanciare esplorazione (scoprire nuove strategie) e sfruttamento (usare le strategie migliori conosciute) è fondamentale in RL per evitare di convergere a soluzioni subottimali.
*   **Interpretazione:**
    *   *Epsilon (DQN):* Per agenti epsilon-greedy come DQN, si monitora il valore di epsilon. Questo tipicamente inizia alto (es. 1.0, pura esplorazione) e decade nel tempo verso un valore basso (es. 0.01-0.1, prevalentemente sfruttamento). Il grafico dovrebbe mostrare questo decadimento programmato. Un decadimento troppo rapido può portare a uno sfruttamento prematuro; troppo lento può ritardare la convergenza.
    *   *Entropia (per policy stocastiche, non DQN standard):* Un'alta entropia all'inizio è buona per l'esplorazione. Dovrebbe diminuire man mano che l'agente diventa più sicuro della policy ottimale. Se rimane troppo alta, l'agente non sta convergendo. Se diventa troppo bassa troppo presto, l'agente potrebbe essere bloccato in un ottimo locale.

### Learning Rate (Tasso di Apprendimento)
*   **Nome Metrica:** Tasso di Apprendimento (Learning Rate)
*   **Cosa mostra:** La dimensione del passo con cui i pesi del modello (es. la rete neurale in DQN) vengono aggiornati durante il training in base all'errore calcolato.
*   **Importanza:** Un iperparametro critico che influenza la velocità e la stabilità della convergenza del modello.
*   **Interpretazione:**
    *   *Fisso:* Il valore rimane costante durante il training.
    *   *Schedulato/Decadente:* Il valore diminuisce nel tempo (es. linearmente, esponenzialmente), spesso per permettere aggiornamenti più grandi all'inizio e più fini verso la fine del training, migliorando la stabilità.
    *   *Valori Tipici:* Dipendono dall'algoritmo, dall'ottimizzatore e dal problema, ma spesso nell'intervallo [1e-5, 1e-2].
    *   Un Tasso di Apprendimento troppo alto può causare divergenza (la perdita aumenta) o oscillazioni attorno all'ottimo. Un Tasso di Apprendimento troppo basso può rendere l'apprendimento molto lento o bloccarlo in ottimi locali. Il suo andamento (se schedulato) va monitorato per assicurarsi che segua il comportamento atteso.

### Value Estimates (Stime di Valore)
*   **Nome Metrica:** Stime di Valore (es. Q-values medi per DQN)
*   **Cosa mostra:** La stima del modello del ritorno atteso (ricompensa cumulativa scontata) per essere in un certo stato (Value function V(s)) o per prendere una certa azione in un certo stato (Action-Value function Q(s,a)).
*   **Importanza:** Indica cosa il modello ha imparato riguardo al valore delle diverse situazioni o azioni. Aiuta a capire se il modello sta discriminando correttamente tra scelte buone e cattive.
*   **Interpretazione:**
    *   *Trend Crescente (per Q-values ottimali):* Generalmente, le stime di valore per gli stati/azioni buoni dovrebbero aumentare nel tempo man mano che l'agente apprende.
    *   *Stabilità:* Dovrebbero stabilizzarsi una volta che la policy converge.
    *   *Scala dei Valori:* La scala assoluta dipende dalla definizione della ricompensa e dal fattore di sconto (gamma).
    *   *Sovrastima/Sottostima:* Alcuni algoritmi (come DQN classico) possono tendere a sovrastimare i Q-values. Monitorare se i valori diventano irrealisticamente alti può essere utile. Tecniche come Double DQN mirano a mitigare questo.
    *   *Per DQN:* Si possono tracciare i Q-values medi per le azioni scelte, i Q-values massimi medi per stato, o la distribuzione dei Q-values.

### TD Error (Errore di Differenza Temporale)
*   **Nome Metrica:** Errore di Differenza Temporale (TD Error)
*   **Cosa mostra:** La differenza (errore) tra la stima corrente del valore di uno stato (o stato-azione) e una stima aggiornata (target) basata sull'esperienza successiva (ricompensa ricevuta + valore scontato dello stato successivo). È il segnale di errore che guida l'apprendimento in molti algoritmi RL basati sul valore.
*   **Importanza:** Indica quanto "sorprendente" è stata una transizione per l'agente. Errori TD elevati (in valore assoluto) guidano maggiori aggiornamenti ai pesi del modello.
*   **Interpretazione:**
    *   *Diminuzione nel Tempo:* Idealmente, l'errore TD medio (o la sua magnitudine) dovrebbe diminuire man mano che le stime di valore dell'agente diventano più accurate e coerenti nel tempo.
    *   *Fluttuazioni:* Normali, specialmente in ambienti stocastici o durante l'esplorazione.
    *   *Valori Alti Persistenti:* Possono indicare difficoltà nell'apprendimento, un ambiente molto imprevedibile, o un learning rate inappropriato.
    *   *Per DQN:* La loss stessa (MSE tra Q-predetto e Q-target) è una forma di errore TD aggregato. Monitorare la loss è quindi un modo diretto per osservare il TD error.

### KL Divergence (Divergenza KL o proxy)
*   **Nome Metrica:** Divergenza KL (Kullback-Leibler) (o un suo proxy)
*   **Cosa mostra:** Misura quanto una distribuzione di probabilità diverge da una seconda distribuzione di probabilità di riferimento. In RL, può essere usata per misurare il cambiamento nella policy dell'agente tra iterazioni di training (es. tra la vecchia policy e la nuova policy dopo un aggiornamento).
*   **Importanza:** Aiuta a monitorare la stabilità dell'apprendimento. Grandi cambiamenti nella policy (alta KL Divergence) possono portare a instabilità e a "dimenticare" quanto appreso precedentemente. Algoritmi come TRPO e PPO usano la KL Divergence per vincolare gli aggiornamenti della policy.
*   **Interpretazione:**
    *   *Valori Bassi e Stabili:* Generalmente desiderabili, indicano che la policy sta cambiando gradualmente e in modo controllato.
    *   *Picchi Elevati:* Possono indicare aggiornamenti drastici alla policy, che potrebbero essere destabilizzanti.
    *   *Proxy per DQN (Variazione di Epsilon):* Per DQN, che ha una policy deterministica (data la stima dei Q-values) più l'esplorazione epsilon-greedy, una vera KL Divergence sulla policy non è tipicamente calcolata. La variazione di epsilon nel tempo è un indicatore di come cambia il bilanciamento esplorazione/sfruttamento, ma è pre-schedulata.
    *   *Proxy per DQN (Variazione dei pesi della rete):* Si potrebbe monitorare la magnitudine degli aggiornamenti ai pesi della rete Q (es. norma della differenza dei pesi tra iterazioni) come un proxy indiretto del cambiamento della policy. Un cambiamento più diretto potrebbe essere la variazione nella distribuzione delle azioni scelte su un set di stati di test.
    *   *Proxy per DQN (Variazione di Epsilon, come menzionato nel task):* Se interpretato come la variazione del parametro epsilon stesso, il suo grafico è deterministico e riflette la schedulazione dell'esplorazione.

### Explained Variance (Varianza Spiegata)
*   **Nome Metrica:** Varianza Spiegata
*   **Cosa mostra:** Misura la proporzione della varianza nei valori target (es. ritorni campionati o target della funzione valore) che è spiegata dalle predizioni del modello di valore (es. V(s) o Q(s,a)). Un valore di 1 significa una predizione perfetta della varianza, 0 significa che il modello non spiega affatto la varianza (equivalente a predire sempre la media dei target). Può essere negativo se il modello è peggiore che predire la media.
*   **Importanza:** Indica quanto bene il modello di valore sta approssimando i veri valori e catturando la loro variabilità. È una misura della "bontà di adattamento" della funzione valore.
*   **Interpretazione:**
    *   *Valori Vicini a 1:* Desiderabile, indica che le stime di valore del modello sono ben correlate con i ritorni effettivi e ne spiegano la varianza.
    *   *Valori Bassi o Negativi:* Il modello di valore non sta apprendendo bene o le stime sono poco accurate.
    *   *Trend Crescente:* Idealmente, dovrebbe aumentare durante il training man mano che il modello apprende.
    *   *Per DQN nel trading:* Potrebbe essere calcolata sulla base di quanto bene i Q-values predetti (o una funzione valore derivata) spiegano la varianza dei ritorni effettivi (o dei target Bellman) osservati.

### Success Rate (Tasso di Successo)
*   **Nome Metrica:** Tasso di Successo
*   **Cosa mostra:** La percentuale di episodi in cui l'agente raggiunge un criterio di successo predefinito.
*   **Importanza:** Una metrica di alto livello e spesso intuitiva per valutare se l'agente sta imparando a risolvere il compito specifico come definito dai criteri di successo.
*   **Interpretazione:**
    *   *Trend Crescente:* Desiderabile, indica che l'agente sta diventando più abile nel raggiungere l'obiettivo.
    *   *Valore Assoluto:* Il target dipende dalla difficoltà del compito e dalla definizione di "successo".
    *   *Definizione del Successo:* È cruciale che sia ben definita e allineata con gli obiettivi generali.
    *   *Per DQN nel trading:* "Successo" potrebbe essere definito come:
        *   Un episodio che termina con un profitto netto positivo.
        *   Raggiungere un certo target di profitto entro l'episodio.
        *   Sopravvivere per l'intera durata dell'episodio senza raggiungere uno stop loss catastrofico.
        *   Superare la performance di un benchmark semplice durante l'episodio.
    *   Un tasso di successo del 70% significa che 7 episodi su 10 soddisfano i criteri definiti.