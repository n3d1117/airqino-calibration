# 1

Salve a tutti, vi presento il lavoro di tesi dal titolo “*Progettazione e sviluppo di componenti per la piattaforma AirQino dedicata al monitoraggio della qualità dell’aria*”, realizzato in collaborazione con l'azienda Magenta e con l’Istituto per la BioEconomia del Consiglio Nazionale delle Ricerche.

# 2

Il monitoraggio della qualità dell’aria è una delle attività più importanti per la tutela della salute pubblica. La qualità dell’aria può essere influenzata da molte sorgenti di emissione, sia naturali che artificiali, tra cui le automobili, le centrali elettriche, gli impianti di riscaldamento, provocando una dispersione nell’aria di inquinanti quali diossido di azoto (NO2), anidride carbonica (CO2), l’ozono (O3) e il particolato atmosferico (PM, in particolare si parla di PM2,5 se il diametro delle particelle non supera i 2,5 nanometri e PM10 se è inferiore ai 10 nanometri). Gli effetti dell’inquinamento atmosferico sulla salute sono molteplici e possono essere a breve o a lungo termine. In questo contesto, i dati raccolti dal monitoraggio della qualità dell’aria possono essere utilizzati per valutare l’impatto dell’inquinamento atmosferico sulla salute umana e sull’ambiente, e per identificare le principali fonti di inquinamento ambientale e per lo sviluppo di piani di mobilità e strategie in tempo reale per il controllo dell’esposizione.

Il monitoraggio della qualità dell’aria in Italia è regolamentato e assegnato alle varie ARPA (Agenzia regionale per la protezione ambientale) e viene effettuato attraverso stazioni fisse, collocate in base a precisi criteri di esposizione, con costose strumentazioni ed elevati costi di manutenzione, fornendo misurazioni dettagliate, che spesso richiedono mesi per la validazione, ma soltanto in posizioni precise e sparse per una determinata città.

In questo scenario si inseriscono i nuovi sensori low-cost per il monitoraggio dell’inquinamento atmosferico integrativo che presentano diversi vantaggi, tra cui il basso costo di installazione, l’alta risoluzione temporale del dato acquisito e l’alta rappresentatività spaziale data dalla presenza capillare di più punti di acquisizione. Una di queste reti è AirQino.

# 3

AirQino è una piattaforma di monitoraggio ambientale ad alta precisione che nasce dall’esigenza di realizzare una rete di stazioni mobile per un monitoraggio più completo della qualità dell’aria in ambito urbano, in linea con la direttiva europee che riconosce e regolamenta l’importanza di misure aggiuntive rispetto a quelle delle stazioni fisse.

Alcune delle caratteristiche di AirQino sono:

- Tutte le stazioni sono configurabili con un’ampia gamma di sensori aggiuntivi a seconda delle proprie esigenze.
- Sistema di rilevamento polifunzionale: il sistema offre la possibilità di rilevare gli agenti inquinanti presenti in atmosfera

La rete di sensori AirQino consente rilevare le principali sostanze inquinanti riscontrabili nell’aria (NO2, O3, CO, PM2.5, PM10) ma anche ulteriori parametri ambientali, come la temperatura, l’umidità relativa dell’aria.

La piattaforma AirQino mette inoltre a disposizione un portale web per la consultazione dei dati, all’indirizzo [https://airqino.magentalab.it](https://airqino.magentalab.it).

La piattaforma consiste in una mappa interattiva che visualizza tutte le reti di centraline. Selezionata una stazione di interesse, vengono mostrati dettagli e foto della stazione stessa (figura 1.5). Per ciascun sensore della centralina selezionata è possibile visualizzare grafici di andamento medio settimanali, insieme al dato istantaneo relativo all’ultima misurazione.

# 4

L’architettura del sistema AirQino è organizzata nei seguenti elementi principali:

- Gateway: server che espone servizi compatibili con le centraline, e normalizza i dati trasmessi verso il server di raccolta. produce anche un output su protocollo MQTT2, utilizzando un broker open source per pubblicare i dati verso il backend.
- Backend: applicazione server che si interfaccia con il broker MQTT e scrive sul database i dati, esponendo inoltre servizi web di tipo REST utilizzati dal frontend web
- Frontend: applicazione web che permette la visualizzazione di mappa e grafici dei dati raccolti dalle centraline, utilizzando i servizi esposti dal backend

# 5

Oggetto di questo studio sono stati due sensori con principi di funzionamento diversi:

- MiCS-2714 per NO2 è un sensore di tipo MOS, ovvero costituito da un film depositato su una piastra di elementi riscaldanti la cui temperatura operativa è generalmente compresa tra 300°C e 500°C. Il costo è contenuto, meno di 5 euro, e la risposta del sensore viene convertita da analogico a digitale con un convertitore con uscita a 10 bit, ovvero un range di 0-1024 counts presente nella scheda di acquisizione
- Il sensore SDS011 per PM basa il suo funzionamento sul principio della diffusione ottica (detta anche dispersione o scattering). In questo caso c’è una precalibrazione di fabbrica e l’uscita è direttamente in unità ingegneristica, ovvero microgrammi per metro cubo, e il costo è di circa 25 euro

# 6

Gli obiettivi sono stati molteplici:

• Sviluppi tecnologici alla piattaforma, rivolti a migliorare in un caso l’affidabilità dei dati inviati dai sensori, e nell’altro la gestione del problema della quantità di dati in aumento costante;

• Studio e analisi del processo di calibrazione delle centraline AirQino, in particolare con un confronto quantitativo sulle diverse tecniche utilizzate per la rilevazione di relazioni tra il segnale dei sensori e la stazione di riferimento;

• Realizzazione di un’interfaccia che permetta la calibrazione di più centraline contemporaneamente.

# 7

Riguardo gli sviluppi realizzati dal punto di vista tecnologico che sono andati direttamente ad impattare la piattaforma AirQino, il primo è stato la realizzazione di una replica del database primario contenente le misurazioni dei sensori. Questo consiste nel copiare e distribuire tutti i dati su un nodo secondario, sempre sincronizzato. Questo porta a diversi vantaggi, tra cui l’alleggerimento del nodo primario perché è possibile effettuare query onerose e test di performance esclusivamente sulla replica.

Il concetto di replica va anche a beneficio dell’affidabilità dei dati delle misurazioni, alla sicurezza e alla ridondanza (essendo distribuiti su più nodi).

# 8

Per la gestione della replicazione è stata utilizzata la tecnologia chiamata streaming replication per database Postgres.

Questa si basa sulle transazioni WAL (Write Ahead Log) e utilizza il protocollo TCP per garantire una connessione sicura tra i server, e inviare in modo asincrono le modifiche al database via via che vengono effettuate. La replica è di sola lettura, ma all’esigenza è possibile promuoverla a nodo primario (es. in caso di guasti).

L’intero setup è stato automatizzato con Docker in modo da mantenere la replica sincronizzata all’avvio del container.

# 9

Come accennato in precedenza, una delle funzionalità della piattaforma AirQino è quella di mostrare per ciascuna centralina la media oraria delle misurazioni nell’ultima settimana. Per fare questo c’è bisogno di fare aggregazione di dati. Il database di produzione conta oltre 100 milioni di misurazioni rilevate, in continuo aumento, con una media di 300 inserimenti al minuto. Da questo punto di vista è necessario garantire efficienza e scalabilità al sistema. Uno dei miglioramenti su questo è stato quello di applicare un sistema di aggregazione continua. È una funzionalità specifica del database utilizzato, Timescale, che consente di automatizzare l’aggregazione di dati a intervalli regolari con aggiornamento automatico, tramite la creazione di una vista materializzata che in questo caso contiene sempre le medie orarie precalcolate.

# 10

Oltre al risparmio di spazio c’è stato un miglioramento significativo delle performance, come si può vedere dai grafici seguenti (prima e dopo). Sono le medie dei tempi di risposta (in secondi) delle query sul database che estraggono le medie orarie di NO2 da tutte le centraline AirQino per l’ultima settimana. Se prima si aveva caricamento anche di 30 secondi per le centraline più attive, dopo l’ottimizzazione i tempi sono nell’ordine dei millisecondi, e soprattutto costanti e indipendenti dalla quantità dei dati.

# 11

Passando invece alla parte di calibrazione, è stata presa in considerazione la centralina SMART16 di AirQino in località Capannori, in provincia di Lucca. Questa rientra nel progetto di monitoraggio Carilucca e dal 2020 è stata in co-locazione con la stazione regionale fissa gestita da ARPAT.

La località di Capannori è di particolare interesse essendo uno dei comuni con più inquinamento atmosferico di tutta la Toscana, data la presenza di una grande quantità di fonti eterogenee di emissione quali l’autostrada, impianti di combustione delle cartiere e l’uso estensivo di riscaldamento domestico.

Con il termine calibrazione si intende analizzare la risposta dei sensori AirQino in base a diverse concentrazioni di inquinanti (prima fase che si fa in laboratorio) e poi sfruttare la vicinanza con la stazione ARPAT di rilevamento per compararli con le misurazioni ufficiali.

# 12

Su questo tema è stato svolta la seguente procedura per i sensori di NO2, PM2.5 e PM10:

- creazione di un dataset in base ai dati della centralina smart16 e ai dati ufficiali ARPAT della sede di Capannori
- allineamento temporale e ricampionamento in base a frequenze specifiche (medie orarie, medie ogni otto ore)
- Analisi grafica dei dati, sia tramite scatterplot che analizzando i residui per individuare relazioni di linearità tra il segnale del sensore e il segnale di riferimento ARPAT
- Sono stati poi applicati diversi modelli di regressione, sia lineare che non lineare, per ottenere delle curve che meglio approssimano la relazione tra i due segnali, sia su tutto il dataset che mese per mese
- Valutazione della performance dei modelli in termini di R^2 (coefficiente di determinazione) e radice dello scarto quadratico medio (RMSE)
- Infine, validazione del modello migliore ottenuto su un set di dati nuovi.

# 13

Qui sono stati riportati i risultati ottenuti con i vari modelli di regressione sui sensori di NO2 e PM2.5, in blu R2 e in arancione RMSE.

Come si può notare il modello di regressione migliore è risultato in media quello lineare avanzato con riconoscimento e rimozione di *outlier*.

In generale, i risultati per i PM2.5 e PM10 sono risultati migliori rispetto alla calibrazione dei sensori di NO2. Questo si può ricondurre in parte alla diversa frequenza dei dati (medie orarie per NO2, medie a otto ore per i PM che riducono la rumorosità dei dati), e in parte anche alla natura e alle differenze dei sensori stessi, questi infatti hanno un principio fisico di funzionamento diverso (SDS011 per i PM è ad infrarossi, più costoso e con output calibrato di fabbrica, mentre il MiCS-2714 per NO2 funziona per ossidazione e richiede calibrazione).

# 14

distanza di cook è una stima dell’influenza di una osservazione in un dataset: è una metrica utile per stimare di quanto cambierebbe un modello di regressione nel caso in cui venga rimossa l’i-esima osservazione. la distanza di cook dell’osservazione i-esima è definita come nella formula, dove:

- n è il numero di osservazioni
- yj è la risposta del modello (valore predetto), yj(i) è la risposta ottenuta escludendo l’i-esima osservazione
- p è il numero di coefficienti della regressione
- MSE è lo scarto quadratico medio

la soglia di cut-off utilizzata, oltre la quale un dato può essere considerato un outlier, è 4 sul numero di osservazioni totali.

Essendo sensori low cost le misurazioni sono spesso molto rumorose, e quindi ha senso applicare modelli che siano robusti rispetto a queste osservazioni *anomale* anche in un'ottica di predizione di dati futuri.

# 15

La corretta convalida di un modello sviluppato per prevedere nuove osservazioni dovrebbe sempre implicare una fase di validazione fatta sul campo (field validation).

Una delle procedure per validare un modello di regressione consiste nel testarlo su nuove osservazioni.

Per svolgere la fase di validazione è stato quindi considerato il periodo dal 01/09/21 al 20/11/21,

in figura è riportato il confronto tra l’andamento di PM2.5 in questo periodo: in grigio le misurazioni grezze del sensore della centralina SMART16, in rosso le stesse misurazioni calibrate con il modello descritto prima, in blu il segnale di riferimento.

gran parte delle osservazioni ARPAT PM2.5 e PM10 vengono sottostimate dai sensori AirQino non calibrati. Questa discrepanza viene tuttavia ridotta significativamente dal processo di calibrazione che in molti casi migliora le performance del sensore in termini di correlazione.

Non abbiamo ancora potuto farla per NO2 perché per adesso non abbiamo dati a disposizione

# 16

Infine è stata realizzata una interfaccia web per rendere più semplice la calibrazione di più centraline allo stesso tempo, salvando più coefficienti insieme: questa operazione veniva prima effettuata singolarmente per ogni sensore di ogni centralina.

Il servizio viene alimentato tramite il caricamento di file csv con struttura nota e preconcordata. I risultati vengono mostrati sullo schermo in forma di tabella, che permette di fare anche delle statistiche avanzate sui coefficienti medi.

# 17

L’interfaccia presenta un colore diverso per ogni riga in base all’esito (ad. esempio rosso in caso di errore con descrizione).

L’applicazione inoltre presenta funzionalità avanzate quali l’autenticazione tramite keycloak solo ad utenti autorizzati, ed è stata sviluppata a microservizi (frontend e backend) con rilascio automatizzato e isolamento delle dipendenze tramite docker.

# 18

Per concludere, lo studio effettuato in questo lavoro di tesi ha evidenziato come le reti di sensori low-cost per il monitoraggio della qualità dell’aria, come AirQino, possano rappresentare una soluzione efficace per il rilevamento dell’inquinamento atmosferico ad alta risoluzione sia temporale che spaziale, e soprattutto

I risultati ottenuti mostrano che anche con sensori a basso costo è possibile misurare inquinanti come NO2, PM2.5 e PM10, che per essere definito tale deve stare sotto una certa soglia di costo

In generale, i risultati per i PM2.5 e PM10 sono risultati di gran lunga migliori rispetto alla calibrazione dei sensori di NO2. La difficoltà di utilizzare sensori low cost per i gassosi è ampiamente dimostrata anche in letteratura. Inoltre questo tipo di sensori possono essere soggetti a contaminazione incrociata con altri gas, che interferiscono sia con la sensibilità che con la precisione della misurazione.

L’applicazione di tecniche di regressione robusta in fase di calibrazione ha riportato miglioramenti significativi in termini di accuratezza.

# 19

Uno sviluppo futuro potrebbe essere quello di perfezionare il processo di calibrazione delle centraline, andando ad introdurre ulteriori parametri quali temperatura e umidità, che da uno studio preliminare hanno dimostrato avere determinate correlazione con gli inquinanti.

Inoltre, si potrebbe prevedere la realizzazione di una procedura di allerta sul segnale delle centraline, analizzando in tempo reale l’andamento del segnale e monitorando la differenza con il riferimento, segnalando quando lo scarto tra i due non è più costante (ad esempio per l’usura del sensore).

in questa ottica si può fare anche post processing analysis (cioè andare a rivedere lo storico per vedere se c'è stata qualche deriva dei sensori segnalando la necessità di una ricalibrazione)

# 20

È tutto, e grazie per l’attenzione!

