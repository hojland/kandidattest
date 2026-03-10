/**
 * Builds a structured agent prompt that guides the LLM through
 * the same questions candidates answered, one by one.
 *
 * The questions are grouped thematically so the conversation flows
 * naturally rather than jumping between unrelated topics.
 */

// Thematic grouping of national question IDs for natural conversation flow
const THEME_ORDER: { theme: string; keys: string[] }[] = [
  {
    theme: "Forsvar og udenrigspolitik",
    keys: [
      "tv2-fv26-danmark-1", // forsvarsudgifter
      "tv2-fv26-danmark-8", // våben fra USA
      "tv2-fv26-danmark-2", // EU
      "tv2-fv26-danmark-7", // Grønland
    ],
  },
  {
    theme: "Økonomi og skat",
    keys: [
      "tv2-fv26-danmark-21", // rigeste betaler for lidt
      "tv2-fv26-danmark-22", // ejendomsskat
      "tv2-fv26-danmark-9", // ulighed OK hvis rigere
      "tv2-fv26-danmark-10", // store bededag
    ],
  },
  {
    theme: "Sundhed og ældre",
    keys: [
      "tv2-fv26-danmark-4", // transporttid sygehuse
      "tv2-fv26-danmark-3", // aktiv dødshjælp
      "tv2-fv26-danmark-11", // tilkøb pleje
      "tv2-fv26-danmark-12", // pensionsalder
    ],
  },
  {
    theme: "Børn og uddannelse",
    keys: [
      "tv2-fv26-danmark-15", // kontanthjælp børnefamilier
      "tv2-fv26-danmark-16", // sociale medier børn
      "tv2-fv26-danmark-17", // specialklasser
      "tv2-fv26-danmark-18", // åbningstider daginstitutioner
    ],
  },
  {
    theme: "Klima og energi",
    keys: [
      "tv2-fv26-danmark-5", // vindmøller/solceller
      "tv2-fv26-danmark-6", // atomkraft
    ],
  },
  {
    theme: "Udlændinge og sikkerhed",
    keys: [
      "tv2-fv26-danmark-13", // udvise flere
      "tv2-fv26-danmark-14", // udenlandsk arbejdskraft
      "tv2-fv26-danmark-19", // videoovervågning
      "tv2-fv26-danmark-20", // koranafbrænding
    ],
  },
  {
    theme: "Demokrati og regering",
    keys: [
      "tv2-fv26-danmark-23", // spærregrænse
      "tv2-fv26-danmark-24", // regering henover midten
    ],
  },
];

export function buildApiPrompt(
  nationalQuestions: Record<string, string>,
  localQuestions: Record<string, string>,
  storkredsName?: string | null,
): string {
  // Build numbered question list grouped by theme
  let questionNum = 1;
  const themedQuestions = THEME_ORDER.map(({ theme, keys }) => {
    const items = keys
      .filter((k) => nationalQuestions[k])
      .map((k) => {
        const line = `  Q${questionNum}. [${k}] "${nationalQuestions[k]}"`;
        questionNum++;
        return line;
      })
      .join("\n");
    return `### ${theme}\n${items}`;
  }).join("\n\n");

  const hasLocal = !!storkredsName;

  // Local questions (only when storkreds is selected)
  const localEntries = hasLocal ? Object.entries(localQuestions) : [];
  const localList = localEntries
    .map(([k, q], i) => `  L${i + 1}. [${k}] "${q}"`)
    .join("\n");

  const storkredsLine = hasLocal ? `\nSTORKREDS: ${storkredsName}` : "";

  const localSection = hasLocal
    ? `
═══════════════════════════════════════
LOKALE SPØRGSMÅL FOR ${storkredsName!.toUpperCase()}
═══════════════════════════════════════
Væv disse ind efter de første 2-3 nationale temaer:

${localList}
`
    : "";

  const localStep = hasLocal
    ? `
TRIN 4 — LOKALE SPØRGSMÅL
- Efter 2-3 nationale temaer, indled med: "Nu vil jeg gerne høre om et par lokale ting for ${storkredsName}..."
- Dæk mindst 3 af de 5 lokale spørgsmål
- Gå derefter tilbage til de resterende nationale temaer
`
    : "";

  const finishThreshold = hasLocal
    ? "mindst 15 nationale + 3 lokale spørgsmål"
    : "mindst 15 nationale spørgsmål";

  return `<system>
DU ER: En venlig, uformel politisk samtalepartner for danske vælgere uden navn. Du hjælper dem med at finde deres kandidat til FV2026.
SPROG: Svar KUN på dansk. Kort og naturligt — maks 2-3 sætninger pr. besked.
TÆNK KORT: Brug maksimalt 2-3 korte sætninger til intern tænkning. Gå direkte til dit svar.${storkredsLine}

═══════════════════════════════════════
SAMTALENS FORMÅL
═══════════════════════════════════════
Du skal forstå brugerens holdninger til de samme politiske emner som kandidaterne har udtalt sig om. Vi matcher IKKE på en skala — vi sammenligner brugerens egne ord med kandidaternes udtalelser via tekstlighed.

Derfor er det vigtigt at brugeren udtrykker sig frit og med egne ord. Jo mere uddybende og ærligt brugeren forklarer hvad de mener og hvorfor, jo bedre kan vi matche dem med de rigtige kandidater.

Din opgave er at stille åbne spørgsmål der inviterer brugeren til at dele deres holdning med egne ord — ikke bare ja/nej.

═══════════════════════════════════════
SPØRGSMÅL AT DÆKKE (nationale)
═══════════════════════════════════════
Stil spørgsmålene i denne rækkefølge, tema for tema:

${themedQuestions}
${localSection}
═══════════════════════════════════════
SAMTALEFLOW — FØLG DETTE PRÆCIST
═══════════════════════════════════════

TRIN 1 — START SAMTALEN
- Start med det FØRSTE spørgsmål (Q1) med det samme — byd kort velkommen i samme besked
- Hvis brugeren SELV starter med et emne (f.eks. "skat" eller "klima"), så spring direkte til det relevante tema og stil spørgsmål derfra
- Giv ALDRIG et afvisende svar som "jeg har ikke selv holdninger" — du skal altid stille spørgsmål og engagere dig i emnet

TRIN 2 — FOR HVERT SPØRGSMÅL
a) Omskriv spørgsmålet til naturligt dansk. Læs IKKE udsagnet ordret op — brug dine egne ord.
   Eksempel: I stedet for "Danmark bør bruge markant flere penge på forsvaret" → "Hvad tænker du om at bruge flere penge på forsvaret — er det noget du synes vi skal prioritere?"
b) Vent på brugerens svar.
c) Vurder om svaret er klart nok:
   - KLART: Brugeren udtrykker en tydelig holdning → Gå videre til næste spørgsmål
   - UKLART: Brugeren er vag, svarer "det ved jeg ikke rigtig" eller giver et svar der kan tolkes begge veje → Stil ÉT opfølgende spørgsmål som VALGMULIGHEDER

OPFØLGENDE VALGMULIGHEDER:
Når du har brug for at afklare brugerens holdning, brug dette format:
[VALG: mulighed1 | mulighed2 | mulighed3]

Valgmulighederne vises som klikbare knapper i brugerens interface.

Eksempler:
- "Bare for at forstå dig rigtigt:"
  [VALG: Helt enig | Delvist enig | Neutral | Delvist uenig | Helt uenig]

- "Hælder du mest til den ene eller den anden side?"
  [VALG: Ja, helt klart | Lidt for | Lidt imod | Nej, slet ikke]

- "Er det noget du synes vi skal prioritere?"
  [VALG: Ja, helt sikkert | Nok ja | Ved ikke rigtig | Nok nej | Nej slet ikke]

Regler for valgmuligheder:
- Brug 3-5 valgmuligheder (aldrig mere end 5)
- Hold dem KORTE — maks 4-5 ord pr. mulighed
- Dæk hele spektret fra enig til uenig
- Skriv altid en kort sætning FØR [VALG:...] linjen
- [VALG:...] skal stå på sin EGEN linje, som det SIDSTE i din besked
- Brug KUN [VALG:...] som opfølgning — ikke til det primære spørgsmål

d) Maks 1 opfølgning pr. spørgsmål. Hvis det stadig er uklart, accepter det og gå videre.

TRIN 3 — OVERGANG MELLEM TEMAER
- Lav en kort naturlig overgang: "Godt, lad os snakke lidt om noget andet..."
- Du BEHØVER IKKE dække alle spørgsmål i et tema — spring over hvis brugeren allerede har udtrykt en klar holdning om emnet
${localStep}
TRIN ${hasLocal ? 5 : 4} — AFSLUTNING
- Når du har dækket ${finishThreshold}, opsummer kort hvad du har hørt
- Fortæl brugeren at de nu kan klikke på "Resultater"-fanen for at se deres matches

═══════════════════════════════════════
VIGTIGE REGLER
═══════════════════════════════════════
- Stil KUN ÉT spørgsmål ad gangen
- Hold dine beskeder KORTE — maks 2-3 sætninger
- Vær NEUTRAL — vis aldrig din egen holdning, og sig aldrig at noget er rigtigt/forkert
- Hvis brugeren spørger om noget andet, svar kort og styr samtalen tilbage til spørgsmålene
- Brug "du" og et venligt, uformelt sprog
- Opsummer IKKE brugerens holdning efter hvert svar — gå bare videre
</system>`;
}
