using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class PlayerControls : MonoBehaviour
{
    public Orb orbPrefab;
    public Image revolverGUI;
    public Image castGUI;
    public Image hpValue;
    public Image horrorValue;

    public Text weaponName;
    public Text loadedAmmo;
    public Text ammoCarried;

    public float valueBarScaleX = 0.8f;

    private float playersHealth, playersSanity;
    private int revolverAmmo;

    // Start is called before the first frame update
    void Start()
    {
        castGUI.enabled = false;
        UpdateWeaponDisplay();
    }

    // Update is called once per frame
    void Update()
    {
        AdjustBars();
        //UpdateWeaponDisplay();

        if (revolverAmmo != Player.revolverAmmo)
        {
            revolverAmmo = Player.revolverAmmo;
            UpdateWeaponDisplay();
        }

        if (Input.GetMouseButtonDown(0))
        {
            switch (Player.GetEquipedWeapon())
            {
                case Player.Weapon.revolver:
                    FireRevolver();
                    break;

                case Player.Weapon.spell:
                    CastSpell();
                    break;
            }           
        }

        if (Input.GetKeyDown(KeyCode.Escape))
            Application.Quit();

        if (Input.GetKey(KeyCode.R) && Player.GetEquipedWeapon() == Player.Weapon.revolver)
        {
            ReloadRevolver();
        }

        float scrollDelta = Input.mouseScrollDelta.y;
        if (scrollDelta != 0)
        {
            SwitchWeapon(scrollDelta);
        }
    }

    public void AdjustBars()
    {
        if (Player.health != playersHealth)
        {
            playersHealth = Player.health;
            float newScaleX = (Player.health / Player.maxHealth) * valueBarScaleX;
            hpValue.transform.localScale = new Vector3(newScaleX, hpValue.transform.localScale.y, hpValue.transform.localScale.z);
        }

        if (Player.sanity != playersSanity)
        {
            UpdateWeaponDisplay();
            playersSanity = Player.sanity;
            float newScaleX = (Player.sanity / Player.maxSanity) * valueBarScaleX;
            horrorValue.transform.localScale = new Vector3(newScaleX, horrorValue.transform.localScale.y, horrorValue.transform.localScale.z);
        }        
    }

    public void UpdateWeaponDisplay()
    {     
        switch (Player.GetEquipedWeapon())
        {
            case Player.Weapon.revolver:
                weaponName.text = "Revolver";
                loadedAmmo.text = "" + Player.bulletsInCylinder + "/" + Player.cylinderSize;
                ammoCarried.text = "" + Player.revolverAmmo + "/" + Player.maxRevolverAmmo;
                break;
            case Player.Weapon.spell:
                weaponName.text = "Azure Flame";
                loadedAmmo.text = "";
                ammoCarried.text = "" + Player.sanity + "/" + Player.maxSanity;
                break;
        }
    }

    public void SwitchWeapon(float delta)
    {
        AnimatorStateInfo revolverAnimState = revolverGUI.GetComponent<Animator>().GetCurrentAnimatorStateInfo(0);
        AnimatorStateInfo spellAnimState = castGUI.GetComponent<Animator>().GetCurrentAnimatorStateInfo(0);

        if (spellAnimState.IsName("CastIdle") && revolverAnimState.IsName("RevolverIdle"))
        {
            castGUI.enabled = false;
            revolverGUI.enabled = false;
            if (delta > 0)
                Player.NextWeapon();
            else
                Player.PrevWeapon();

            switch (Player.GetEquipedWeapon())
            {
                case Player.Weapon.revolver:
                    revolverGUI.enabled = true;
                    break;
                case Player.Weapon.spell:
                    castGUI.enabled = true;
                    break;
            }
            UpdateWeaponDisplay();
        }           
    }

    public void SpawnOrb(Vector3 position, Vector3 direction)
    {
        Orb orb = Instantiate(orbPrefab, position, Quaternion.identity);
        orb.direction = direction;
    }

    public void FireRevolver()
    {
        Animator revolverAnimator = revolverGUI.GetComponent<Animator>();
        AnimatorStateInfo revolverAnimState = revolverAnimator.GetCurrentAnimatorStateInfo(0);

        if (revolverAnimState.IsName("RevolverIdle") && Player.bulletsInCylinder > 0)
        {
            Player.bulletsInCylinder -= 1;
            revolverAnimator.SetTrigger("fire");

            GameObject camera = GameObject.Find("PlayerCamera");
            Vector3 direction = camera.transform.forward.normalized;
            Vector3 position = camera.transform.position;

            if (Physics.Raycast(position, direction, out RaycastHit target))
            {
                if (target.collider.gameObject.tag == "Enemy")
                    target.collider.GetComponent<Enemy>().Damage(Player.revolverDamage);
            }
            UpdateWeaponDisplay();
        }       
    }

    public void ReloadRevolver()
    {
        Animator revolverAnimator = revolverGUI.GetComponent<Animator>();
        AnimatorStateInfo revolverAnimState = revolverAnimator.GetCurrentAnimatorStateInfo(0);

        if (revolverAnimState.IsName("RevolverIdle") && Player.revolverAmmo > 0 && Player.bulletsInCylinder < Player.cylinderSize)
        {
            revolverAnimator.SetTrigger("reload");
            int delta = Player.cylinderSize - Player.bulletsInCylinder;
            if (Player.revolverAmmo >= delta)
            {
                Player.bulletsInCylinder += delta;
                Player.revolverAmmo -= delta;
            }
            else
            {
                Player.bulletsInCylinder += Player.revolverAmmo;
                Player.revolverAmmo = 0;
            }
            UpdateWeaponDisplay();
        }
    }

    public void CastSpell()
    {
        Animator spellAnimator = castGUI.GetComponent<Animator>();
        AnimatorStateInfo spellAnimState = spellAnimator.GetCurrentAnimatorStateInfo(0);

        if (spellAnimState.IsName("CastIdle") && Player.sanity > Player.spellHorror)
        {
            Player.Traumatise(Player.spellHorror);
            spellAnimator.SetTrigger("cast");

            GameObject camera = GameObject.Find("PlayerCamera");
            Vector3 direction = camera.transform.forward.normalized;
            Vector3 position = camera.transform.position;

            SpawnOrb(position + direction, direction);
            UpdateWeaponDisplay();
        }
    }
}
