package be.kuleuven.pylos;

import be.kuleuven.pylos.battle.BattleMT;
import be.kuleuven.pylos.battle.BattleResult;
import be.kuleuven.pylos.battle.data.PlayedGame;
import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.player.PylosPlayer;
import be.kuleuven.pylos.player.PylosPlayerType;
import be.kuleuven.pylos.player.codes.PylosPlayerBestFit;
import be.kuleuven.pylos.player.codes.PylosPlayerMiniMax;
import be.kuleuven.pylos.player.student.*;
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
        for (int i = 1; i <= 10; i++) {

            // 1) run alle battles en verzamel BattleResults
        List<BattleResult> battleResults = runAllBattles();

        // 2) alle PlayedGame's samenvoegen in één lijst
        List<PlayedGame> allGames = new ArrayList<>();
        for (BattleResult br : battleResults) {
            allGames.addAll(br.playedGames);
        }

        System.out.println("Total played games over all battles: " + allGames.size());
        String exportPathWithIndex = EXPORT_PATH.replace(".json", "_" + i + ".json");
        // 3) alles samen in één JSON-bestand steken
        exportGames(allGames, exportPathWithIndex);

        System.out.println("Exported to: " + exportPathWithIndex);
        }
    }

    /**
     * Deze methode:
     *  - definieert meerdere spelers (met verschillende parameters),
     *  - laat alle (gewenste) combinaties tegen elkaar spelen,
     *  - en geeft een lijst van BattleResult terug.
     */
    public static List<BattleResult> runAllBattles() {
        // Definieer je spelers


        PylosPlayerType mm6 = new PylosPlayerType("mm6") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(6);
            }
        };

        PylosPlayerType Tano = new PylosPlayerType("Tano") {
            @Override
            public PylosPlayer create() {
                return new StudentPlayerTano();
            }
        };
        PylosPlayerType Radi = new PylosPlayerType("radi") {
            @Override
            public PylosPlayer create() {
                return new StudentPlayerRadi();
            }
        };
        PylosPlayerType Wannes = new PylosPlayerType("wannes") {
            @Override
            public PylosPlayer create() {
                return new StudentWannes();
            }
        };

        PylosPlayerType ons = new PylosPlayerType("onze") {
            @Override
            public PylosPlayer create() {
                return new StudentPlayer();
            }
        };
        PylosPlayerType random = new PylosPlayerType("random") {
            @Override
            public PylosPlayer create() {
                return new StudentPlayerRandomFit();
            }
        };
        PylosPlayerType Thomas = new PylosPlayerType("Thomas") {
            @Override
            public PylosPlayer create() {
                return new StudentPlayerThomas();
            }
        };
        PylosPlayerType BestFit = new PylosPlayerType("BestFit") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerBestFit();
            }
        };



        // Als je er later nog wil bijsteken, gewoon hier toevoegen
        List<PylosPlayerType> players = Arrays.asList( BestFit, random,  mm6, Tano, Radi, Wannes, Thomas, ons);


        int gamesPerMatchup = 500;
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



