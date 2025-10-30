package be.kuleuven.pylos.main;

import be.kuleuven.pylos.battle.Battle;
import be.kuleuven.pylos.battle.BattleMT;
import be.kuleuven.pylos.battle.BattleResult;
import be.kuleuven.pylos.battle.RoundRobin;
import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosGame;
import be.kuleuven.pylos.game.PylosGameObserver;
import be.kuleuven.pylos.player.PylosPlayer;
import be.kuleuven.pylos.player.PylosPlayerObserver;
import be.kuleuven.pylos.player.PylosPlayerType;
import be.kuleuven.pylos.player.codes.PlayerFactoryCodes;
import be.kuleuven.pylos.player.codes.PylosPlayerBestFit;
import be.kuleuven.pylos.player.codes.PylosPlayerMiniMax;
import be.kuleuven.pylos.player.student.StudentPlayer;
import be.kuleuven.pylos.player.student.StudentPlayer2;
import be.kuleuven.pylos.player.student.StudentPlayerRandomFit;

import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.io.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
public class PylosMain {

    public static void main(String[] args) {
        /* !!! jvm argument !!! -ea */

        //startSingleGame();
        //startBattle();
        startBattleMultithreaded();
        //startRoundRobinTournament();
    }

    public static void startSingleGame() {

        Random random = new Random(0);

        PylosPlayer playerLight = new PylosPlayerBestFit();
        PylosPlayer playerDark = new PylosPlayerMiniMax(4);

        PylosBoard pylosBoard = new PylosBoard();
        PylosGame pylosGame = new PylosGame(pylosBoard, playerLight, playerDark, random, PylosGameObserver.CONSOLE_GAME_OBSERVER, PylosPlayerObserver.NONE);

        pylosGame.play();
    }

    public static void startBattle() {
        int nRuns = 500;
        PylosPlayerType p1 = new PylosPlayerType("Student") {
            @Override
            public PylosPlayer create() {

                return new StudentPlayer();
                // return new PylosPlayerBestFit();
            }
        };

        PylosPlayerType p2 = new PylosPlayerType("Minimax4") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(4);
            }
        };

        Battle.play(p1, p2, nRuns);
    }

    // --- NIEUW: Constanten en helpers voor logging en hervatten ---
    private static final String LOG_FILE_NAME = "tuning_results.log";

    // Regex om de *laatst voltooide* run te vinden (matcht zowel succes als falen)
    private static final Pattern PROGRESS_PATTERN = Pattern.compile(
            "\\[(\\d+)/\\d+\\] Params\\(H:(\\d+), S:(\\d+), A:(\\d+), T:(\\d+), R:(\\d+)\\) ->");

    // Regex om het *beste* resultaat tot nu toe te vinden
    private static final Pattern BEST_PATTERN = Pattern.compile(
            "!!! NEW BEST: (.*?)%% with H:(\\d+), S:(\\d+), A:(\\d+), T:(\\d+), R:(\\d+)");

    /**
     * Helper-klasse om de staat opgeslagen in het logbestand te bewaren.
     */
    private static class ResumeState {
        double bestWinRate = -1.0;
        int[] bestParams = new int[5];
        int[] lastCompletedParams = null; // H, S, A, T, R
        int lastCombinationCount = 0;
    }
    // --- EINDE NIEUWE HELPERS ---


    /**
     * AANGEPASTE METHODE: Start de battle met logging en hervat-logica.
     */
    public static void startBattleMultithreaded() {
        int nRuns = 100; // Runs per parameter set
        int nThreads = 8;
        final double MAX_TIME_PER_GAME_MS = 1000.0;

        // --- Hyperparameter Tuning Setup ---
        int[] heightScores =    {0, 50, 100, 150, 250 };
        int[] squareScores =    {0, 50, 100, 150 ,250};
        int[] availableScores = {0, 50, 100, 150 ,250 };
        int[] trappedScores =   {0, 50, 100, 150 ,250 };
        int[] reserveScore = {0, 50, 100, 150 ,250 };

        // --- NIEUW: Laad vorige staat (indien aanwezig) ---
        ResumeState state = parseLogForResume();
        double bestWinRate = state.bestWinRate;
        int[] bestParams = state.bestParams;
        int[] lastCompletedParams = state.lastCompletedParams;
        boolean foundLastCompleted = (lastCompletedParams == null); // true als we *niet* hervatten (alles draaien)

        // --- Define the constant opponent ---
        PylosPlayerType p2 = new PylosPlayerType("Minimax4") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(4);
            }
        };

        int totalCombinations = heightScores.length * squareScores.length * availableScores.length * trappedScores.length * reserveScore.length;

        // --- NIEUW: Open logbestand in 'append' modus ---
        // try-with-resources zorgt ervoor dat de writer netjes gesloten wordt
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(LOG_FILE_NAME, true))) {

            // --- NIEUW: Log start- of hervatbericht ---
            if (foundLastCompleted) {
                log(writer, "Start nieuwe massive hyperparameter tuning...");
                log(writer, "Test " + totalCombinations + " combinaties (all non-negative) tegen Minimax4.");
            } else {
                log(writer, "---------------------------------------------------------");
                log(writer, "--- HERVATTEN van hyperparameter tuning ---");
                log(writer, "Hervat na combinatie " + state.lastCombinationCount + "/" + totalCombinations);
                log(writer, String.format("Laatst voltooide parameters: H:%d, S:%d, A:%d, T:%d, R:%d",
                        lastCompletedParams[0], lastCompletedParams[1], lastCompletedParams[2], lastCompletedParams[3], lastCompletedParams[4]));
                log(writer, String.format("Huidige beste win rate: %.2f%%", bestWinRate * 100));
            }
            log(writer, "Constraint: Gemiddelde tijd moet < " + (MAX_TIME_PER_GAME_MS / 1000.0) + " sec/game zijn.");
            log(writer, "---------------------------------------------------------");


            int combinationCount = 0; // Live teller

            // 4. Start de grid search loop
            for (int h : heightScores) {
                for (int s : squareScores) {
                    for (int a : availableScores) {
                        for (int t : trappedScores) {
                            for (int r : reserveScore) {
                                combinationCount++; // Verhoog de teller voor *elke* combinatie

                                // --- NIEUW: Sla reeds geteste combinaties over ---
                                if (!foundLastCompleted) {
                                    // Vergelijk met de laatst voltooide set
                                    if (h == lastCompletedParams[0] &&
                                            s == lastCompletedParams[1] &&
                                            a == lastCompletedParams[2] &&
                                            t == lastCompletedParams[3] &&
                                            r == lastCompletedParams[4]) {
                                        foundLastCompleted = true; // Dit was de laatste, verwerk de *volgende*
                                    }
                                    continue; // Sla deze combinatie over (ofwel is het te vroeg, ofwel is het de laatste)
                                }

                                // --- Vanaf hier is de code identiek, behalve voor logging ---

                                final int currentH = h;
                                final int currentS = s;
                                final int currentA = a;
                                final int currentT = t;
                                final int currentR = r;

                                // 1. Create the P1 player type
                                String playerName = String.format("Student(H:%d,S:%d,A:%d,T:%d, R:%d)", h, s, a, t, r);
                                PylosPlayerType p1 = new PylosPlayerType(playerName) {
                                    @Override
                                    public PylosPlayer create() {
                                        // Gebruik je StudentPlayer2 constructor (deze is al in scope)
                                        return new StudentPlayer2(currentH, currentS, currentA, currentT, currentR);
                                    }
                                };

                                // 2. Run the battle
                                // BattleMT is al in scope via je imports
                                BattleResult result = BattleMT.play(p1, p2, nRuns, nThreads, false);

                                // 3. Check time constraint
                                double avgTimePerGameMs = (double) result.runTime / nRuns;

                                if (avgTimePerGameMs >= MAX_TIME_PER_GAME_MS) {
                                    String failMsg = String.format("[%d/%d] Params(H:%d, S:%d, A:%d, T:%d, R:%d) -> FAILED: Time %.2f sec/game. Skipping.",
                                            combinationCount, totalCombinations, h, s, a, t, r, avgTimePerGameMs / 1000.0);
                                    log(writer, failMsg); // Gebruik log-helper
                                    continue;
                                }

                                // 4. Get the win rate
                                int p1Wins = result.p1Wins();
                                int totalGames = nRuns;
                                double winRate = (totalGames == 0) ? 0 : (double) p1Wins / totalGames;

                                // 5. Print progress
                                String progressMsg = String.format("[%d/%d] Params(H:%d, S:%d, A:%d, T:%d, R:%d) -> Win Rate: %.2f%% (%d/%d) | Time: %.2f sec/game",
                                        combinationCount, totalCombinations, h, s, a, t, r, winRate * 100, p1Wins, totalGames, avgTimePerGameMs / 1000.0);
                                log(writer, progressMsg); // Gebruik log-helper

                                // 6. Track the best result
                                if (winRate > bestWinRate) {
                                    bestWinRate = winRate;
                                    bestParams[0] = h;
                                    bestParams[1] = s;
                                    bestParams[2] = a;
                                    bestParams[3] = t;
                                    bestParams[4] = r;

                                    String bestMsg = String.format("!!! NEW BEST: %.2f%% with H:%d, S:%d, A:%d, T:%d, R:%d !!!",
                                            bestWinRate * 100, h, s, a, t, r);
                                    log(writer, bestMsg); // Gebruik log-helper
                                }
                            }
                        }
                    }
                }
            }

            // 7. Report the final best result
            log(writer, "---------------------------------------------------------");
            log(writer, "Hyperparameter tuning finished.");

            if (bestWinRate == -1.0) {
                log(writer, "Geen enkele parameter set voldeed aan de tijdslimiet van < " + (MAX_TIME_PER_GAME_MS / 1000.0) + " sec/game.");
            } else {
                log(writer, "Best parameters found (meeting time constraints):");
                log(writer, String.format("  Height Score:    %d", bestParams[0]));
                log(writer, String.format("  Square Score:    %d", bestParams[1]));
                log(writer, String.format("  Available Score: %d", bestParams[2]));
                log(writer, String.format("  Trapped Score:   %d", bestParams[3]));
                log(writer, String.format("  Reserve Score:   %d", bestParams[4])); // VERBETERD: Reserve score toegevoegd
                log(writer, String.format("  Best Win Rate vs Minimax4: %.2f%%", bestWinRate * 100));
            }
            log(writer, "---------------------------------------------------------");

        } catch (IOException e) {
            // Fout bij het openen/schrijven van het logbestand
            System.err.println("CRITICAL ERROR: Kon logbestand " + LOG_FILE_NAME + " niet openen om te schrijven.");
            e.printStackTrace();
        }
    }

    /**
     * NIEUWE METHODE: Parset het logbestand om de laatste staat te vinden.
     */
    private static ResumeState parseLogForResume() {
        ResumeState state = new ResumeState();
        File logFile = new File(LOG_FILE_NAME);

        if (!logFile.exists()) {
            return state; // Geen logbestand, begin vanaf nul
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(logFile))) {
            String line;
            while ((line = reader.readLine()) != null) {

                // Zoek naar de laatst gelogde voortgang
                Matcher progressMatcher = PROGRESS_PATTERN.matcher(line);
                if (progressMatcher.find()) {
                    if (state.lastCompletedParams == null) {
                        state.lastCompletedParams = new int[5];
                    }
                    state.lastCombinationCount = Integer.parseInt(progressMatcher.group(1));
                    state.lastCompletedParams[0] = Integer.parseInt(progressMatcher.group(2)); // H
                    state.lastCompletedParams[1] = Integer.parseInt(progressMatcher.group(3)); // S
                    state.lastCompletedParams[2] = Integer.parseInt(progressMatcher.group(4)); // A
                    state.lastCompletedParams[3] = Integer.parseInt(progressMatcher.group(5)); // T
                    state.lastCompletedParams[4] = Integer.parseInt(progressMatcher.group(6)); // R
                }

                // Zoek naar het laatst gelogde beste resultaat
                Matcher bestMatcher = BEST_PATTERN.matcher(line);
                if (bestMatcher.find()) {
                    state.bestWinRate = Double.parseDouble(bestMatcher.group(1)) / 100.0;
                    state.bestParams[0] = Integer.parseInt(bestMatcher.group(2)); // H
                    state.bestParams[1] = Integer.parseInt(bestMatcher.group(3)); // S
                    state.bestParams[2] = Integer.parseInt(bestMatcher.group(4)); // A
                    state.bestParams[3] = Integer.parseInt(bestMatcher.group(5)); // T
                    state.bestParams[4] = Integer.parseInt(bestMatcher.group(6)); // R
                }
            }
        } catch (IOException e) {
            System.err.println("Waarschuwing: Kon logbestand niet lezen om te hervatten. Start opnieuw. Fout: " + e.getMessage());
            return new ResumeState(); // Ga door met een schone lei
        } catch (NumberFormatException e) {
            System.err.println("Waarschuwing: Fout bij parsen van logbestand. Start opnieuw. Fout: " + e.getMessage());
            return new ResumeState(); // Ga door met een schone lei
        }

        return state;
    }

    /**
     * NIEUWE METHODE: Helper om naar console ÉN bestand te loggen.
     */
    private static void log(BufferedWriter writer, String message) {
        try {
            // 1. Log naar console
            System.out.println(message);

            // 2. Log naar bestand
            if (writer != null) {
                writer.write(message + "\n");
                writer.flush(); // Zorg ervoor dat het *direct* wordt geschreven
            }
        } catch (IOException e) {
            System.err.println("Kritieke fout bij schrijven naar logbestand: " + e.getMessage());
        }
    }
    public static void startRoundRobinTournament() {
        //Same requirements apply as for startBattleMultithreaded()

        //Create your own PlayerFactory containing all PlayerTypes you want to test
        PlayerFactoryCodes pFactory = new PlayerFactoryCodes();
        //PlayerFactoryStudent pFactory = new PlayerFactoryStudent();

        int nRunsPerCombination = 1000;
        int nThreads = 8;

        Set<RoundRobin.Match> matches = RoundRobin.createTournament(pFactory);

        RoundRobin.play(matches, nRunsPerCombination, nThreads);

        List<BattleResult> results = matches.stream().map(c -> c.battleResult).collect(Collectors.toList());

        RoundRobin.printWinsMatrix(results, pFactory.getTypes());

    }
}
