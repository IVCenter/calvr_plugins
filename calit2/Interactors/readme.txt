<ChannelConfig>
  <Channel width="1920" height="1080" left="0" bottom="0" comment="Main" channel="0" stereoMode="QUAD_BUFFER"  windowIndex="0"  name="0"  /> 
</ChannelConfig> 
            
<WindowConfig> 
  <Window width="1920" comment="Main" window="0" pipeIndex="0" height="1080" left="0" bottom="0" name="0" decoration="false" quadBuffer="true" /> 
</WindowConfig> 
           
<ScreenConfig> 
  <Screen width="521" comment="Main" h="0" originX="0" originY="0" originZ="0" height="292" p="-60.0" r="0.0" name="0" screen="0" />
</ScreenConfig>          

<SceneSize value="521" />
<ViewerPosition x="0" y="-450" z="300" />
<EyeSeparation value="on" />

<NumPipes value="1" />
<NumScreens value="1" />
<NumWindows value="1" />
<Stereo value="true" />
<Stereo separation="4" />
<Near value="10" />


<Plugin>
  <Interactors value="on" />
</Plugin>

<Input>
  <TrackingSystem0 value="VRPN" >
    <Orientation h="0" p="90" r="0" />
    <Offset x="0" y="-230" z="350" />
    <NumBodies value="2" />
    <NumButtons value="3" />
    <VRPN>
      <Server value="Tracker0@skarn.ucsd.edu" />
    </VRPN>
    <Body0>  <!-- Head -->
      <Orientation h="0" p="0" r="0" />
    </Body0>
    <Body1>   <!-- Hand -->
      <Orientation h="0" p="-90" r="0" />
      <Offset x="0" y="0" z="0" />
    </Body1>
  </TrackingSystem0>

  <NumHeads value="1" />
  <NumHands value="1" />
  <Head0Address system="0" body="0" />
  <Hand0>
    <Address system="0" body="1" />
    <ButtonMask system0="0xff" system1="0x00" />
  </Hand0>
</Input>

 <MenuSystem type="BOARDMENU" useHints="false">
    <BoardMenu>
      <Position distance="300" />
      <Scale value="0.3" />
      <Buttons select="0" open="1" />
      <Trigger value="DOUBLECLICK" />
    </BoardMenu>
 </MenuSystem>
