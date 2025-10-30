package be.kuleuven.pylos;

import be.kuleuven.pylos.battle.BattleMT;
import be.kuleuven.pylos.battle.BattleResult;
import be.kuleuven.pylos.battle.data.PlayedGame;
import be.kuleuven.pylos.player.PylosPlayer;
import be.kuleuven.pylos.player.PylosPlayerType;
import be.kuleuven.pylos.player.codes.PylosPlayerBestFit;
import be.kuleuven.pylos.player.codes.PylosPlayerMiniMax;
import com.google.gson.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;


public class PylosMLCollect2 {

    public static final String EXPORT_PATH = "pylos-ml/src/main/training/resources/games/0.json";
    public static final int N_THREADS = 8; // Number of threads for BattleMT

    public static void main(String[] args) throws IOException {
        System.out.println("Collecting diverse set of games...");

        // Collect games
        List<PlayedGame> playedGames = PylosMLCollect.collectGames();

        System.out.println("Total collected games: " + playedGames.size());

        // Export to json file
        File file = new File(EXPORT_PATH);
        Files.createDirectories(file.getParentFile().toPath());
        FileWriter writer = new FileWriter(file);
        Gson gson = new GsonBuilder().create();

        System.out.println("Exporting to JSON file: " + EXPORT_PATH);
        gson.toJson(playedGames, writer);

        writer.flush();
        writer.close();

        System.out.println("Export complete.");
    }

    /**
     * Collects a diverse set of 100,000 games by pitting PylosPlayerBestFit
     * against various depths of PylosPlayerMiniMax.
     *
     * @return A list containing 100,000 PlayedGame objects.
     */
    public static List<PlayedGame> collectGames() {

        // List to hold all games from all battles.
        // Initialize with 100,000 capacity for efficiency.
        List<PlayedGame> allPlayedGames = new ArrayList<>(100000);

        // 1. Define the common player: BestFit
        PylosPlayerType p1_BF = new PylosPlayerType("BF") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerBestFit();
            }
        };

        // 2. Define MM2 and run battle (10,000 games)
        PylosPlayerType p2_MM2 = new PylosPlayerType("MM2") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(2);
            }
        };
        System.out.println("Starting battle 1/4: BF vs MM2 (10,000 games)...");
        BattleResult br1 = BattleMT.play(p1_BF, p2_MM2, 10000, N_THREADS, true);
        allPlayedGames.addAll(br1.playedGames);
        System.out.println("Finished battle 1. Total games collected: " + allPlayedGames.size());

        // 3. Define MM4 and run battle (15,000 games)
        PylosPlayerType p2_MM4 = new PylosPlayerType("MM4") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(4);
            }
        };
        System.out.println("Starting battle 2/4: BF vs MM4 (15,000 games)...");
        BattleResult br2 = BattleMT.play(p1_BF, p2_MM4, 15000, N_THREADS, true);
        allPlayedGames.addAll(br2.playedGames);
        System.out.println("Finished battle 2. Total games collected: " + allPlayedGames.size());

        // 4. Define MM6 and run battle (20,000 games)
        PylosPlayerType p2_MM6 = new PylosPlayerType("MM6") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(6);
            }
        };
        System.out.println("Starting battle 3/4: BF vs MM6 (20,000 games)...");
        BattleResult br3 = BattleMT.play(p1_BF, p2_MM6, 20000, N_THREADS, true);
        allPlayedGames.addAll(br3.playedGames);
        System.out.println("Finished battle 3. Total games collected: " + allPlayedGames.size());

        // 5. Define MM8 and run battle (55,000 games)
        PylosPlayerType p2_MM8 = new PylosPlayerType("MM8") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(8);
            }
        };
        System.out.println("Starting battle 4/4: BF vs MM8 (55,000 games)...");
        BattleResult br4 = BattleMT.play(p1_BF, p2_MM8, 55000, N_THREADS, true);
        allPlayedGames.addAll(br4.playedGames);
        System.out.println("Finished battle 4. Total games collected: " + allPlayedGames.size());

        // 6. Return the combined list
        return allPlayedGames;
    }
}
