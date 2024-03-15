within TestLib;

model boundary_test
  Modelica.Blocks.Sources.CombiTimeTable Boundaries(columns = 2:3, fileName = Modelica.Utilities.Files.loadResource("modelica://TestLib/bound_file.txt"), tableName = "table1", tableOnFile = true) annotation(
    Placement(visible = true, transformation(origin = {-70, 10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation

  annotation(
    uses(Modelica(version = "4.0.0")));
end boundary_test;
