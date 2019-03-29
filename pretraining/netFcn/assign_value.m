function a = assign_value(a, i, b, val)
    if iscell(val(1))
        a.params(i(1)).(b) = cell2mat(val(1));
        a.params(i(2)).(b) = cell2mat(val(2));
    else
        a.params(i(1)).(b) = val(1);
        a.params(i(2)).(b) = val(2);
    end
end