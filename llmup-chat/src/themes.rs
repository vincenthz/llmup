use gpui::{Action, App, SharedString};
use gpui_component::{ActiveTheme, Theme, ThemeMode, ThemeRegistry, scroll::ScrollbarShow};

pub fn init(cx: &mut App) {
    if let Some(theme) = ThemeRegistry::global(cx)
        .themes()
        .get("Default Light")
        .cloned()
    {
        Theme::global_mut(cx).apply_config(&theme);
    }

    cx.observe_global::<Theme>(|cx| {
        // when theme change
    })
    .detach();

    cx.on_action(|switch: &SwitchTheme, cx| {
        let theme_name = switch.0.clone();
        if let Some(theme_config) = ThemeRegistry::global(cx).themes().get(&theme_name).cloned() {
            Theme::global_mut(cx).apply_config(&theme_config);
        }
        cx.refresh_windows();
    });
    cx.on_action(|switch: &SwitchThemeMode, cx| {
        let mode = switch.0;
        Theme::change(mode, None, cx);
        cx.refresh_windows();
    });

    cx.refresh_windows();
}

#[derive(Action, Clone, PartialEq)]
#[action(namespace = themes, no_json)]
pub(crate) struct SwitchTheme(pub(crate) SharedString);

#[derive(Action, Clone, PartialEq)]
#[action(namespace = themes, no_json)]
pub(crate) struct SwitchThemeMode(pub(crate) ThemeMode);
