package be.kuleuven.pylos;

import be.kuleuven.pylos.battle.BattleMT;
import be.kuleuven.pylos.battle.BattleResult;
import be.kuleuven.pylos.battle.data.PlayedGame;
import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.player.PylosPlayer;
import be.kuleuven.pylos.player.PylosPlayerType;
import be.kuleuven.pylos.player.codes.PylosPlayerBestFit;
import be.kuleuven.pylos.player.codes.PylosPlayerMiniMax;
import com.google.gson.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class PylosMLCollect {

    public static final String EXPORT_PATH = Paths.get("pylos-ml", "src", "main", "training", "resources", "games", "all_battles.json").toString();


    public static void main(String[] args) throws IOException {
        // 1) run alle battles en verzamel BattleResults
        List<BattleResult> battleResults = runAllBattles();

        // 2) alle PlayedGame's samenvoegen in één lijst
        List<PlayedGame> allGames = new ArrayList<>();
        for (BattleResult br : battleResults) {
            allGames.addAll(br.playedGames);
        }

        System.out.println("Total played games over all battles: " + allGames.size());

        // 3) alles samen in één JSON-bestand steken
        exportGames(allGames, EXPORT_PATH);
    }

    /**
     * Deze methode:
     *  - definieert meerdere spelers (met verschillende parameters),
     *  - laat alle (gewenste) combinaties tegen elkaar spelen,
     *  - en geeft een lijst van BattleResult terug.
     */
    public static List<BattleResult> runAllBattles() {
        // Definieer je spelers
        PylosPlayerType mm3_1 = new PylosPlayerType("MM_4_1") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(4);
            }
        };

        PylosPlayerType mm3_2 = new PylosPlayerType("MM_4_2") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(4);
            }
        };


        PylosPlayerType mm4_1 = new PylosPlayerType("mm4_1") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(4);
            }
        };

        PylosPlayerType mm4_2 = new PylosPlayerType("mm4_2") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(4);
            }
        };

        PylosPlayerType mm2_1 = new PylosPlayerType("mm2_1") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(2);
            }
        };

        PylosPlayerType mm5_1 = new PylosPlayerType("mm5_1") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(5);
            }
        };

        PylosPlayerType mm5_2 = new PylosPlayerType("mm5_2") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(5);
            }
        };

        PylosPlayerType mm6 = new PylosPlayerType("mm6") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(6);
            }
        };



        // Als je er later nog wil bijsteken, gewoon hier toevoegen
        List<PylosPlayerType> players = Arrays.asList(mm3_1, mm3_2, mm4_1, mm4_2, mm2_1, mm5_1, mm5_2, mm6);

        int gamesPerMatchup = 3000;   // of 10000, wat jij wil
        int threads = 8;
        boolean recordGames = true;

        List<BattleResult> results = new ArrayList<>();

        // Laat elke speler tegen elke andere spelen
        for (int i = 0; i < players.size(); i++) {
            for (int j = i; j < players.size(); j++) { // i<j voor alleen verschillende; i<=j voor ook mirror/self
                PylosPlayerType p1 = players.get(i);
                PylosPlayerType p2 = players.get(j);

                System.out.println("Starting battle: " + p1.toString() + " vs " + p2.toString());

                BattleResult br = BattleMT.play(p1, p2, gamesPerMatchup, threads, recordGames);
                results.add(br);

                System.out.println("Finished battle: " + p1.toString() + " vs " + p2.toString() +
                        " -> games: " + br.playedGames.size());
            }
        }

        return results;
    }

    /**
     * Schrijft een lijst van PlayedGame naar één JSON-bestand.
     */
    private static void exportGames(List<PlayedGame> games, String exportPath) throws IOException {
        File file = new File(exportPath);
        Files.createDirectories(file.getParentFile().toPath());

        try (FileWriter writer = new FileWriter(file)) {
            Gson gson = new GsonBuilder().create();
            gson.toJson(games, writer);
        }

        System.out.println("Exported to: " + exportPath);
    }
}

/*
    public static List<PlayedGame> collectGames() {
        SavedModelBundle model = SavedModelBundle.load(MODEL_PATH, "serve");
        PylosPlayerType p1 = new PylosPlayerType("PML_1") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerML(model);
            }
        };
        PylosPlayerType p2 = new PylosPlayerType("PML_2") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerML(model);
            }
        };

        BattleResult br = BattleMT.play(p1, p2, 10000, 8, true);

        return br.playedGames;
    }
    */



