using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Content.Interaction;
public class XrOriginCharControllerSettings : MonoBehaviour
{
    public bool continuousTurn = true;
    private LocomotionSystem locomotionSystem;
    private ActionBasedContinuousTurnProvider actionBasedContinuousTurnProvider;

    private ActionBasedSnapTurnProvider actionBasedSnapTurnProvider;
    public         XRLever m_RightHandTurnStyleToggle;
    public         GameObject tunneling;

    // Start is called before the first frame update
    void Start()
    {
        var controller = GetComponent<CharacterController>();
        controller.detectCollisions = false;
        locomotionSystem = GetComponent<LocomotionSystem>();
        actionBasedContinuousTurnProvider  = GetComponent<ActionBasedContinuousTurnProvider>();
        actionBasedSnapTurnProvider  = GetComponent<ActionBasedSnapTurnProvider>();

        //SetTurnType(true);
        m_RightHandTurnStyleToggle.onLeverDeactivate.AddListener(SetTurnTypeSnap);// to the red, move it to the left
        m_RightHandTurnStyleToggle.onLeverActivate.AddListener(SetTurnTypeCont); //to the green, move it to the right
    
        //force it to set some value to prevent duplicate input
        if (m_RightHandTurnStyleToggle.value)
            SetTurnTypeCont();
        else
            SetTurnTypeSnap();

    }

    // Update is called once per frame
    void Update()
    {
        // only for debug!
        //if(Input.GetKeyDown(KeyCode.L))
        //   SwitchTurnType();
    }

    public void SwitchTurnType()
    {
        if (continuousTurn)
        {
            //if was continous disable it and enable snap
            SetTurnType(false);
        }
        else
        {
            //if was snap disable it and enable continuous
            SetTurnType(true);
        }
        
    }

    void SetTurnType(bool cont = false)

    {
        actionBasedContinuousTurnProvider.enabled = cont;
        actionBasedSnapTurnProvider.enabled = !cont;
        continuousTurn = cont;
    }
    
    void SetTurnTypeCont()
    {
        SetTurnType(true);
    } 
    void SetTurnTypeSnap()
    {
        SetTurnType();
    }
    
    public void SwitchComfortTunnel()
    {
        tunneling.SetActive(!tunneling.activeSelf);
        
    }

}

