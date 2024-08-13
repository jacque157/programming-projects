package Weapons;

import javafx.scene.input.MouseEvent;

/**
 * Interface represents weapon which player can use.
 */
public interface Weapon
{
    /**
     * @return the damage dealt to enemy.
     */
    double getDamage();

    /**
     * @return the name of weapon.
     */
    String getName();

    /**
     * @return max ammo which can be carried.
     */
    int getMaxAmmo();

    /**
     * @return ammo loaded in gun.
     */
    int getAmmoLoaded();

    /**
     * @return maximum ammo loaded.
     */
    int getClipSize();

    /**
     * Hide weapon from GUI.
     */
    void hide();

    /**
     * Show weapon in GUI.
     */
    void show();

    /**
     * @return whether the weapon can be used.
     */
    boolean isReady();

    /**
     * Plays appropriate animations and resolves attacking.
     * @param event mouse event
     */
    void fire(MouseEvent event);

    /**
     * Plays appropriate animation and reloads weapon.
     */
    void reload();

    /**
     * @return string representing information about loaded munition.
     */
    String munitionLoadedInfo();

    /**
     * @return string representing information about munition carried.
     */
    String munitionInfo();
}
