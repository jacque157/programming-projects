package Game;

import javafx.geometry.Pos;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;

import java.io.File;

/**
 * Static class which handles graphical user interface.
 */
public class GUI
{
    static public final Font PANEL_FONT = Font.font("Malgun Gothic Semilight", FontWeight.BOLD, 19);
    static private final GridPane panel = new GridPane();
    static private final BorderPane layout = new BorderPane();
    static private final Pane screen = new Pane();

    static private Label lbHealth;
    static private Label lbSanity;

    static private Label lbActiveWeapon;
    static private Label lbAmmoLoaded;
    static private Label lbAmmoCarried;


    /**
     * @return panel which holds player's HUD.
     */
    public static GridPane getPanel() { return panel; }

    /**
     * @return borderPane of application which holds both scene and player's HUD.
     */
    public static BorderPane getLayout() { return layout; }

    /**
     * @return screen representing 3d game.
     */
    public static Pane getScreen() { return screen; }

    /**
     * @return Player's health label.
     */
    public static Label getLbHealth() { return lbHealth; }

    /**
     * @return Player's sanity label.
     */
    public static Label getLbSanity() { return lbSanity; }

    /**
     * @return Player's active weapon label.
     */
    public static Label getLbActiveWeapon() { return lbActiveWeapon; }

    /**
     * @return Amount of bullets loaded in weapon.
     */
    public static Label getLbAmmoLoaded() { return lbAmmoLoaded; }

    /**
     * @return Amount of bullets carried by player.
     */
    public static Label getLbAmmoCarried() { return lbAmmoCarried; }


    /**
     * Initializes GUI.
     */
    public static void init()
    {
        layout.setCenter(screen);

        GridPane bottomPane = new GridPane();
        bottomPane.add(panel, 0, 0);
        layout.setBottom(bottomPane);

        screen.getChildren().add(Renderer.getSubScene());
        initBottomPanel();
    }

    /**
     * Displays the "You win" screen.
     */
    public static void initWinScreen()
    {
        layout.setBottom(null);
        layout.setCenter(null);

        BackgroundImage background = new BackgroundImage(new Image("Assets" + File.separator + "sprites" + File.separator + "background.png"), BackgroundRepeat.REPEAT, BackgroundRepeat.REPEAT, BackgroundPosition.DEFAULT,
                BackgroundSize.DEFAULT);
        BorderPane winScreen = new BorderPane();
        winScreen.setBackground(new Background(background));
        Label label = new Label("Congratulations you have managed to escape! YOU WIN!");
        label.setTextFill(Color.GOLD);
        label.setFont(Font.font("Malgun Gothic Semilight", FontWeight.BOLD, 25));

        winScreen.setCenter(label);
        layout.setCenter(winScreen);
    }

    /**
     * Displays the "You lose" screen.
     * @param deathByInsanity Whether the cause of losing game was insanity.
     */
    public static void initGameOverScreen( boolean deathByInsanity )
    {
        layout.setBottom(null);
        layout.setCenter(null);

        BackgroundImage background = new BackgroundImage(new Image("Assets" + File.separator + "sprites" + File.separator + "background.png"), BackgroundRepeat.REPEAT, BackgroundRepeat.REPEAT, BackgroundPosition.DEFAULT,
                BackgroundSize.DEFAULT);
        BorderPane winScreen = new BorderPane();
        winScreen.setBackground(new Background(background));
        Label label = new Label( deathByInsanity ? "Your mind could not handle any more horrors. YOU LOSE!" : "You succumbed to your wounds. YOU LOSE!");
        label.setTextFill( deathByInsanity? Color.BLUEVIOLET : Color.CRIMSON);
        label.setFont(Font.font("Malgun Gothic Semilight", FontWeight.BOLD, 25));

        winScreen.setCenter(label);
        layout.setCenter(winScreen);
    }

    private static void initBottomPanel()
    {
        BackgroundImage background = new BackgroundImage(new Image("Assets" + File.separator + "sprites" + File.separator + "background.png"), BackgroundRepeat.REPEAT, BackgroundRepeat.NO_REPEAT, BackgroundPosition.DEFAULT,
                BackgroundSize.DEFAULT);

        panel.setBackground(new Background(background));

        lbHealth = new Label("Vitality: " + Player.getHealth() + " / " + Player.getMaxVitality());
        lbHealth.setTextFill(Color.INDIANRED);
        lbHealth.setFont(PANEL_FONT);
        lbSanity = new Label("Reason: " + Player.getSanity() + " / " + Player.getMaxReason());
        lbSanity.setFont(PANEL_FONT);
        lbSanity.setTextFill(Color.BLUEVIOLET);

        lbActiveWeapon = new Label(Player.getActiveWeapon().getName());
        lbActiveWeapon.setFont(PANEL_FONT);
        lbActiveWeapon.setTextFill(Color.DARKGRAY);
        lbAmmoLoaded = new Label("" + Player.getActiveWeapon().getAmmoLoaded() + " / " + Player.getActiveWeapon().getClipSize());
        lbAmmoLoaded.setFont(PANEL_FONT);
        lbAmmoLoaded.setTextFill(Color.DARKGRAY);
        lbAmmoCarried = new Label("" + Player.getRevolverAmmo() + " / " + Player.getActiveWeapon().getMaxAmmo());
        lbAmmoCarried.setFont(PANEL_FONT);
        lbAmmoCarried.setTextFill(Color.DARKGRAY);

        panel.addColumn(0, lbHealth, lbSanity);
        panel.addColumn(1, lbActiveWeapon, lbAmmoLoaded, lbAmmoCarried);

        //panel.setGridLinesVisible(true);
        panel.setVgap(10);
        panel.setHgap(30);
        panel.setAlignment(Pos.CENTER);
    }

    /**
     * Update's the Weapon info of hud depending of weapon and player's stats.
     */
    public static void updateWeaponGUI()
    {
        GUI.getLbActiveWeapon().setText(Player.getActiveWeapon().getName());
        GUI.getLbAmmoLoaded().setText(Player.getActiveWeapon().munitionLoadedInfo());
        GUI.getLbAmmoCarried().setText(Player.getActiveWeapon().munitionInfo());
    }
}
