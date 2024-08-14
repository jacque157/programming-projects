using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using UnityEngine.XR.Interaction.Toolkit;

public class DialChange : XRBaseInteractable
{
    [SerializeField] [Tooltip("The object that is visually grabbed and manipulated")]
    private Transform m_Handle = null;


    /// <summary>
    /// The object that is visually grabbed and manipulated
    /// </summary>
    public Transform handle
    {
        get => m_Handle;
        set => m_Handle = value;
    }


    private IXRSelectInteractor m_Interactor;

    [SerializeField] [Tooltip("he")] private Vector3 startRot;
    [SerializeField] [Tooltip("Teh")] private Vector3 startPos;
    [SerializeField] [Tooltip("eh")] private Vector3 currentPos;
    [SerializeField] [Tooltip("heh")] private float newAngle;
    [SerializeField] private float angleTest;

    [SerializeField] [Tooltip("multiplies the real position to speed up the dial")]
    private float angleSensitivity = 180f;

    private void Start()
    {
        //SetValue(m_Value);
        //SetKnobRotation(ValueToRotation());
    }

    protected override void OnEnable()
    {
        base.OnEnable();
        //change
        selectEntered.AddListener(StartGrab);
        selectExited.AddListener(EndGrab);
    }

    protected override void OnDisable()
    {
        selectEntered.RemoveListener(StartGrab);
        selectExited.RemoveListener(EndGrab);
        base.OnDisable();
    }

    private void StartGrab(SelectEnterEventArgs args)
    {
        //Debug.Log("StartGrab");
        m_Interactor = args.interactorObject;
        startRot = handle.transform.localEulerAngles;
        startPos = m_Interactor.transform.position;
        UpdateRotation();
    }

    //change
    private void EndGrab(SelectExitEventArgs args)
    {
        //Debug.Log("EndGrab");
        m_Interactor = null;
    }

    public override void ProcessInteractable(XRInteractionUpdateOrder.UpdatePhase updatePhase)
    {
        base.ProcessInteractable(updatePhase);
        if (updatePhase == XRInteractionUpdateOrder.UpdatePhase.Dynamic)
            if (isSelected)
            {
                currentPos = m_Interactor.transform.position;
                UpdateRotation();
            }
    }

    private void UpdateRotation()
    {
        var angle = startRot.y;

        //Vector3 a = ...;
        //Vector3 b = ...;
        var planeNormal = transform.up;
        var projectionA = Vector3.ProjectOnPlane(transform.forward, planeNormal);
        var projectionB = Vector3.ProjectOnPlane(currentPos - startPos, planeNormal);

        angleTest = Vector3.Angle(projectionA, projectionB);


        //we dont want to update rotation if hand deviated too much from the area 
        //get dir from handle to hand 
        var dirH = currentPos - handle.position;
        //get the magnitude
        var mag = dirH.magnitude;
        //TODO test this, maybe its not good etc
        //TODO ak sme blizko tak ta projekcia nestaci 
        //asi to osetrime tak ze ak sme blizko , t.j. velkost projectionB je mala, neriesime angleTest
        //pretoze sme velmi blizko nuly a to kazi angleTest
        if (angleTest < 25.0f || angleTest > 180 - 25 || projectionB.magnitude<0.15f)
        {
            
            //but we want the change in the correct dir only, which is up

            var dir = 1;
            var checkAgainst = transform.forward;
            //check if we are under or upper
            if (angleTest > 90)
            {
                checkAgainst = -transform.forward;
                dir = -1;
            }
            //newAngle = angle - (currentPos.y - startPos.y) * angleSensitivity;
            newAngle = angle - dir * (currentPos - startPos).magnitude * angleSensitivity;
            if (newAngle > 360)
                newAngle %= 360;
            if (newAngle < 0)
                newAngle += 360;
            newAngle = Math.Clamp(newAngle, 0, 360);
            m_Handle.localEulerAngles = new Vector3(m_Handle.localEulerAngles.x,
                newAngle,
                m_Handle.localEulerAngles.z);
        }
    }

    public float GetAngle()
    {
        return newAngle;
    }
}